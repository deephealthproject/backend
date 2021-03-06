import json
import os
import random
from os.path import join as opjoin
from pathlib import Path

import numpy as np
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from celery import shared_task
from pyeddl.tensor import Tensor

from backend import settings
from backend_app import models as dj_models
from deeplearning import bindings
from deeplearning.utils import Logger, dotdict


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
    args = dotdict(args)
    ckpts_dir = opjoin(settings.TRAINING_DIR, 'ckpts')
    outputfile = None
    inference = None
    training = None
    output_dir = None
    net = None
    train = True if args.mode == 'training' else False
    batch_size = args.batch_size if args.mode == 'training' else args.test_batch_size

    weight_id = args.weight_id
    weight = dj_models.ModelWeights.objects.get(id=weight_id)
    pretrained = None
    if train:
        training = dj_models.Training.objects.get(id=args.training_id)
        if weight.pretrained_on:
            pretrained = weight.pretrained_on.location
    else:
        inference_id = args.inference_id
        inference = dj_models.Inference.objects.get(id=inference_id)
        pretrained = weight.location

    logger = Logger()
    if train:
        logger.open(Path(training.logfile), 'w')
    else:
        logger.open(inference.logfile, 'w')
        outputfile = open(inference.outputfile, 'w')
        output_dir = inference.outputfile[:-4] + '.d'
        os.makedirs(output_dir, exist_ok=True)
    # Save args to file
    logger.print_log('args: ' + json.dumps(args, indent=2, sort_keys=True))

    if pretrained:
        net = eddl.import_net_from_onnx_file(pretrained)
    else:
        net = eddl.import_net_from_onnx_file(weight.model_id.location)
    # if train:
    #     size = [args.input_h, args.input_w]  # Height, width
    # else:  # inference
    #     # get size from input layers
    #     size = net.layers[0].input.shape[2:]
    # FIXME EDDL does not allow editing of input layers from onnx
    # -> always use onnx size as input
    size = net.layers[0].input.shape[2:]

    try:
        dataset_path = str(dj_models.Dataset.objects.get(id=args.dataset_id).path)
    except KeyError:
        raise Exception(f'Dataset with id: {args.dataset_id} not found in bindings.py')

    dataset = bindings.dataset_binding.get(args.dataset_id)

    if dataset is None and not train:
        # Binding does not exist. it's a single image dataset
        # Use as dataset "stub" the dataset on which model has been trained
        dataset = bindings.dataset_binding.get(weight.dataset_id.id)
    elif dataset is None and train:
        raise Exception(f'Dataset with id: {args.dataset_id} not found in bindings.py')

    basic_augs = ecvl.SequentialAugmentationContainer([ecvl.AugResizeDim(size)])
    train_augs = basic_augs
    val_augs = basic_augs
    test_augs = basic_augs
    if args.train_augs:
        train_augs = ecvl.SequentialAugmentationContainer([
            ecvl.AugResizeDim(size), ecvl.AugmentationFactory.create(args.train_augs)
        ])
    if args.val_augs:
        val_augs = ecvl.SequentialAugmentationContainer([
            ecvl.AugResizeDim(size), ecvl.AugmentationFactory.create(args.val_augs)
        ])
    if args.test_augs:
        test_augs = ecvl.SequentialAugmentationContainer([
            ecvl.AugResizeDim(size), ecvl.AugmentationFactory.create(args.test_augs)
        ])

    logger.print_log('Reading dataset'', flush=True')
    dataset = dataset(dataset_path, batch_size, ecvl.DatasetAugmentations([train_augs, val_augs, test_augs]))
    d = dataset.d
    num_classes = dataset.num_classes
    # in_ = eddl.Input([d.n_channels_, size[0], size[1]])
    # out_ = eddl.Sigmoid(model(in_, num_classes))
    # net = eddl.Model([in_], [out_])

    if train:
        eddl.build(
            net,
            eddl.adam(args.lr),
            [bindings.losses_binding.get(args.loss)],
            [bindings.metrics_binding.get(args.metric)],
            eddl.CS_GPU([1], mem='low_mem') if args.gpu else eddl.CS_CPU()
        )
    else:  # inference
        eddl.build(
            net,
            o=eddl.adam(args.lr),
            cs=eddl.CS_GPU([1], mem='low_mem') if args.gpu else eddl.CS_CPU(),
            init_weights=False
        )
    net.resize(batch_size)  # resize manually since we don't use "fit"
    eddl.summary(net)

    images = Tensor([batch_size, d.n_channels_, size[0], size[1]])
    if train:
        gts = Tensor([batch_size, d.n_channels_gt_, size[0], size[1]])

    # TODO create gts also in test if they exist

    logger.print_log(f'Starting {args.mode}')
    if train:
        num_samples_train = len(d.GetSplit(ecvl.SplitType.training))
        num_batches_train = num_samples_train // batch_size
        num_samples_val = len(d.GetSplit(ecvl.SplitType.validation))
        num_batches_val = num_samples_val // batch_size

        evaluator = Evaluator()
        indices = list(range(batch_size))
        miou = -1
        for e in range(args.epochs):
            eddl.reset_loss(net)
            d.SetSplit(ecvl.SplitType.training)
            s = d.GetSplit()
            random.shuffle(s)
            d.split_.training_ = s
            d.ResetAllBatches()
            for i in range(num_batches_train):
                d.LoadBatch(images, gts)
                images.div_(255.0)
                gts.div_(255.0)
                eddl.train_batch(net, [images], [gts], indices)

                losses = eddl.get_losses(net)
                metrics = eddl.get_metrics(net)
                logger.print_log(f'Train Epoch: {e + 1}/{args.epochs} [{i + 1}/{num_batches_train}]'
                                 f'{net.losses[0].name}={losses[0]:.3f} - {net.metrics[0].name}={metrics[0]:.3f}')

            if len(d.split_.validation_) > 0:
                logger.print_log(f'Validation {e}/{args.epochs}')

                d.SetSplit(ecvl.SplitType.validation)
                evaluator.ResetEval()
                for j in range(num_batches_val):
                    logger.print_log(f'Val Epoch: {e + 1}/{args.epochs}  [{j + 1}/{num_batches_val}]', end='')
                    d.LoadBatch(images, gts)
                    images.div_(255.0)
                    gts.div_(255.0)
                    eddl.forward(net, [images])
                    output = eddl.getOutput(eddl.getOut(net)[0])
                    for k in range(batch_size):
                        img_ = output.select([str(k)])
                        gts_ = gts.select([str(k)])
                        a, b = np.array(img_, copy=False), np.array(gts_, copy=False)
                        iou = evaluator.BinaryIoU(a, b)
                        logger.print_log(f' - IoU: {iou:.6g}')

                last_miou = evaluator.MIoU()
                logger.print_log(f'Val Epoch: {e + 1}/{args.epochs} - MIoU: {last_miou:.6f}')

                if last_miou > miou:
                    miou = last_miou
                    eddl.save_net_to_onnx_file(net, opjoin(ckpts_dir, f'{weight_id}.onnx'))
                    logger.print_log('Weights saved')

            else:
                eddl.save_net_to_onnx_file(net, opjoin(ckpts_dir, f'{weight_id}.onnx'))
                logger.print_log('Weights saved')
    else:
        d.SetSplit(ecvl.SplitType.test)
        num_samples_test = len(d.GetSplit())
        num_batches_test = num_samples_test // batch_size
        for j in range(num_batches_test):
            logger.print_log(f'Infer Batch {j + 1}/{num_batches_test}')
            d.LoadBatch(images)
            images.div_(255.0)
            eddl.forward(net, [images])
            preds = eddl.getOut(net)[0]

            for k in range(batch_size):
                pred = preds.select([str(k)])
                # gt = gts.select([str(k)])
                # pred_np, gt = np.array(pred, copy=False), np.array(gt, copy=False)
                pred_np = np.array(pred, copy=False)
                # iou = evaluator.BinaryIoU(pred_np, gt)
                # print(f'Inference {batch_size * j + k + 1}/{num_batches * batch_size} IoU: {iou:.6f}', flush=True)
                # logging.info(f'Inference {batch_size * j + k + 1}/{num_batches * batch_size} IoU: {iou:.6f}')
                pred_np[pred_np >= 0.5] = 255
                pred_np[pred_np < 0.5] = 0

                sample_index = d.GetSplit()[j * batch_size + k]
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
        outputfile.close()

        logger.print_log('Inference completed')
    logger.print_log('<done>')
    logger.close()
    del net
    return
