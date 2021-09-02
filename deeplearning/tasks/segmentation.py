import json
import os
import random
from os.path import join as opjoin
from pathlib import Path

import numpy as np
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from celery import shared_task
from pyeddl._core import eddl as eddl_core
from pyeddl.tensor import Tensor

from backend import settings
from deeplearning.utils import FINAL_LAYER, Logger


class Evaluator:
    def __init__(self):
        self.eps = 1e-06
        self.thresh = 0.5
        self.buf = []

    def ResetEval(self):
        self.buf = []

    def BinaryIoU(self, a, b):
        intersection = np.logical_and(a >= self.thresh, b >= self.thresh).sum()
        union = np.logical_or(a >= self.thresh, b >= self.thresh).sum()
        rval = (intersection + self.eps) / (union + self.eps)
        self.buf.append(rval)
        return rval

    def MIoU(self):
        if not self.buf:
            return 0
        return sum(self.buf) / len(self.buf)


@shared_task
def segment(args):
    ckpts_dir = opjoin(settings.TRAINING_DIR, 'ckpts')
    outputfile = None
    output_dir = None

    train = True if args.get('mode') == 'training' else False
    batch_size = args.get('batch_size')
    epochs = args.get('epochs')
    task = args.get('task')
    net = args.get('net')
    layer_to_remove = net.get('layer_to_remove')
    dataset = args.get('dataset')
    weight = args.get('weight')

    logger = Logger()
    logger.open(Path(task.get('logfile')), 'w')
    if not train:
        outputfile = open(task.get('outputfile'), 'w')
        output_dir = task.get('outputfile')[:-4] + '.d'
        os.makedirs(output_dir, exist_ok=True)
    # Save args to file
    logger.print_log('args: ' + json.dumps(args, indent=2, sort_keys=True))

    size = [args.get('input_h'), args.get('input_w')]  # Height, width

    # Define augmentations for splits
    train_augs = None
    val_augs = None
    test_augs = None
    if args.get('train_augs'):
        train_augs = ecvl.SequentialAugmentationContainer([
            ecvl.AugmentationFactory.create(args.get('train_augs'))
        ])
    if args.get('val_augs'):
        val_augs = ecvl.SequentialAugmentationContainer([
            ecvl.AugmentationFactory.create(args.get('val_augs'))
        ])
    if args.get('test_augs'):
        test_augs = ecvl.SequentialAugmentationContainer([
            ecvl.AugmentationFactory.create(args.get('test_augs'))
        ])

    logger.print_log('Reading dataset')
    dataset_path = dataset.get('path')
    ctypes = [eval(dataset.get('ctype'))]
    if dataset.get('ctype_gt'):
        ctypes.append(eval(dataset.get('ctype_gt')))

    dataset_augs = ecvl.DatasetAugmentations([train_augs, val_augs, test_augs])
    d = ecvl.DLDataset(dataset_path, batch_size, dataset_augs, *ctypes, num_workers=args.get('num_workers'))
    num_classes = len(d.classes_)

    net = eddl.import_net_from_onnx_file(net.get('location'), input_shape=[d.n_channels_, *size])

    # Retrieve the name of the input layer
    l_input = None
    for l in net.layers:
        if isinstance(l, eddl_core.LInput):
            l_input = eddl.getLayer(net, l.name)
            break
    assert l_input is not None
    out_ = eddl.getOut(net)[0]

    # Add activation if not in graph
    if not isinstance(out_, eddl_core.LActivation):
        out = eddl.Sigmoid(out_)
        net = eddl.Model([l_input], [out])

    if train:
        if args.get('remove_layer') and layer_to_remove:
            l_ = eddl.getLayer(net, layer_to_remove)
            if num_classes != l_.output.shape[1] and (num_classes != 0 and l_.output.shape[1] != 1):
                # Last layer must be replaced
                net_layer_names = [l.name for l in net.layers]
                layer_to_remove_index = net_layer_names.index(layer_to_remove)
                # Remove all layers from the end to "layer_to_remove"
                ksize = l_.params[0].shape[2:]  # Retrieve kernel size from the conv to remove
                for gts_ in range(len(net_layer_names) - 1, layer_to_remove_index - 1, -1):
                    eddl.removeLayer(net, net_layer_names[gts_])
                top = eddl.getLayer(net, net_layer_names[layer_to_remove_index - 1])
                if d.n_channels_ > 3:
                    out = eddl.Sigmoid(eddl.Conv3D(top, num_classes, ksize, use_bias=True, name=FINAL_LAYER))
                else:
                    out = eddl.Sigmoid(eddl.Conv2D(top, num_classes, ksize, use_bias=True, name=FINAL_LAYER))
                net = eddl.Model([l_input], [out])

        eddl.build(
            net,
            eddl.adam(args.get('lr')),
            [args.get('loss')],
            [args.get('metric')],
            eddl.CS_GPU([1], mem='low_mem') if args.get('gpu') else eddl.CS_CPU(),
            False
        )
        if args.get('remove_layer') and layer_to_remove:
            # Force initialization of new layers
            eddl.initializeLayer(net, FINAL_LAYER)
    else:  # inference
        eddl.build(
            net,
            o=eddl.adam(args.get('lr')),
            cs=eddl.CS_GPU([1], mem='low_mem') if args.get('gpu') else eddl.CS_CPU(),
            init_weights=False
        )

    net.resize(batch_size)  # resize manually since we don't use "fit"
    eddl.summary(net)

    images = Tensor([batch_size, d.n_channels_, size[0], size[1]])
    gts = Tensor([batch_size, d.n_channels_gt_, size[0], size[1]])

    # TODO create gts also in test if they exist

    logger.print_log(f'Starting {args.get("mode")}')
    if train:
        num_batches_train = d.GetNumBatches(ecvl.SplitType.training)
        # Look for validation split
        d_has_validation = any([True for s in d.split_ if s.split_name_ == 'validation'])

        if d_has_validation:
            num_batches_val = d.GetNumBatches(ecvl.SplitType.validation)

        evaluator = Evaluator()
        miou = -1
        for e in range(epochs):
            eddl.reset_loss(net)
            d.SetSplit(ecvl.SplitType.training)
            d.ResetBatch(shuffle=True)

            # Resize to batch size if we have done a previous resize
            if d.split_[d.current_split_].last_batch_ != batch_size:
                # last mini-batch could have different size
                net.resize(batch_size)

            d.Start()
            for b in range(num_batches_train):
                d.LoadBatch(images, gts)
                _, x, y = d.GetBatch()

                # If it's the last batch and the number of samples doesn't fit the batch size, resize the network
                if b == num_batches_train - 1 and x.shape[0] != batch_size:
                    # last mini-batch could have different size
                    net.resize(x.shape[0])
                eddl.train_batch(net, [x], [y])

                losses = eddl.get_losses(net)
                metrics = eddl.get_metrics(net)
                logger.print_log(f'Train - epoch [{e + 1}/{epochs}] - batch [{b + 1}/{num_batches_train}]'
                                 f' - loss={losses[0]:.3f} - metric={metrics[0]:.3f}')
            d.Stop()
            if d_has_validation:
                eddl.reset_loss(net)
                logger.print_log(f'Validation {e}/{epochs}')
                d.SetSplit(ecvl.SplitType.validation)
                d.ResetBatch()
                evaluator.ResetEval()
                d.Start()
                for b in range(num_batches_val):
                    _, x, y = d.GetBatch()

                    # If it's the last batch and the number of samples doesn't fit the batch size, resize the network
                    if b == num_batches_train - 1 and x.shape[0] != batch_size:
                        # last mini-batch could have different size
                        net.resize(x.shape[0])

                    eddl.forward(net, [x])
                    output = eddl.getOutput(out_)
                    logger.print_log(f'Validation - epoch [{e + 1}/{epochs}] - batch [{b + 1}/{num_batches_val}]',
                                     end='')
                    for bs in range(batch_size):
                        img_ = np.array(output.select([str(bs)]), copy=False)
                        gts_ = np.array(y.select([str(bs)]), copy=False)
                        iou = evaluator.BinaryIoU(img_, gts_)
                        logger.print_log(f' - IoU: {iou:.3f}', end='')
                    logger.print_log('')
                d.Stop()
                last_miou = evaluator.MIoU()
                logger.print_log(f'Validation - epoch [{e + 1}/{epochs}] - MIoU: {last_miou:.3f}')

                if last_miou > miou:
                    miou = last_miou
                    eddl.save_net_to_onnx_file(net, opjoin(ckpts_dir, f'{weight.get("id")}.onnx'))
                    logger.print_log('Weights saved')
            else:
                eddl.save_net_to_onnx_file(net, opjoin(ckpts_dir, f'{weight.get("id")}.onnx'))
                logger.print_log('Weights saved')
    else:
        eddl.set_mode(net, 0)

        d.SetSplit(ecvl.SplitType.test)
        num_batches_test = d.GetNumBatches(ecvl.SplitType.test)
        d.ResetAllBatches()
        out_layer = eddl.getOut(net)[0]

        # Resize to batch size if we have done a previous resize
        if d.split_[d.current_split_].last_batch_ != batch_size:
            # last mini-batch could have different size
            net.resize(batch_size)
        d.Start()
        for b in range(num_batches_test):
            _, x, _ = d.GetBatch()

            # If it's the last batch and the number of samples doesn't fit the batch size, resize the network
            if b == num_batches_test - 1 and x.shape[0] != batch_size:
                # last mini-batch could have different size
                net.resize(x.shape[0])
            eddl.forward(net, [x])

            logger.print_log(f'Inference - batch [{b + 1}/{num_batches_test}]')
            output = eddl.getOutput(out_layer)

            for bs in range(batch_size):
                pred = output.select([str(bs)])
                # gt = gts.select([str(k)])
                # pred_np, gt = np.array(pred, copy=False), np.array(gt, copy=False)
                pred_np = np.array(pred, copy=False)
                # iou = evaluator.BinaryIoU(pred_np, gt)
                # print(f'Inference {batch_size * j + k + 1}/{num_batches * batch_size} IoU: {iou:.6f}', flush=True)
                # logging.info(f'Inference {batch_size * j + k + 1}/{num_batches * batch_size} IoU: {iou:.6f}')
                pred_np[pred_np >= 0.5] = 255
                pred_np[pred_np < 0.5] = 0

                sample_index = d.GetSplit()[b * batch_size + bs]
                orig_image_path = d.samples_[sample_index].location_[0]
                orig_image_name = Path(orig_image_path).name.split('.')
                orig_image_name = orig_image_name[0] + '.png'

                # Save image to file
                img_ecvl = ecvl.TensorToImage(pred)
                if num_classes == 1:
                    img_ecvl.colortype_ = ecvl.ColorType.GRAY
                else:
                    img_ecvl.colortype_ = ecvl.ColorType.BGR
                img_ecvl.channels_ = "xyc"
                # Convert to original size
                ecvl.ResizeDim(img_ecvl, img_ecvl, d.samples_[sample_index].size_)

                ecvl.ImWrite(opjoin(output_dir, orig_image_name), img_ecvl)
                outputfile.write(opjoin(output_dir, orig_image_name) + '\n')
        d.Stop()
        outputfile.close()

        logger.print_log('Inference completed')
    logger.print_log('<done>')
    logger.close()
    del net
    return
