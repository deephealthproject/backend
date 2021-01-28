import json
import random
from os.path import join as opjoin
from pathlib import Path

import numpy as np
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from celery import shared_task
from pyeddl.tensor import Tensor

from backend import settings
from deeplearning import bindings
from deeplearning.utils import Logger


@shared_task
def classificate(args):
    ckpts_dir = opjoin(settings.TRAINING_DIR, 'ckpts')
    outputfile = None

    train = True if args.get('mode') == 'training' else False
    batch_size = args.get('batch_size')
    epochs = args.get('epochs')
    task = args.get('task')
    net = args.get('net')
    dataset = args.get('dataset')
    weight = args.get('weight')

    logger = Logger()
    logger.open(Path(task.get('logfile')), 'w')
    if not train:
        outputfile = open(task.get('outputfile'), 'w')

    # Save args to file
    logger.print_log('args: ' + json.dumps(args, indent=2, sort_keys=True))

    net = eddl.import_net_from_onnx_file(net.get('location'))

    # if train:
    #     size = [args.input_h, args.input_w]  # Height, width
    # else:  # inference
    #     # get size from input layers
    #     size = net.layers[0].input.shape[2:]

    # FIXME EDDL does not allow editing of input layers from onnx
    # -> always use onnx size as input
    size = net.layers[0].input.shape[2:]

    # Define augmentations for splits
    basic_augs = ecvl.SequentialAugmentationContainer([ecvl.AugResizeDim(size)])
    train_augs = basic_augs
    val_augs = basic_augs
    test_augs = basic_augs
    if args.get('train_augs'):
        train_augs = ecvl.SequentialAugmentationContainer([
            ecvl.AugResizeDim(size), ecvl.AugmentationFactory.create(args.get('train_augs'))
        ])
    if args.get('val_augs'):
        val_augs = ecvl.SequentialAugmentationContainer([
            ecvl.AugResizeDim(size), ecvl.AugmentationFactory.create(args.get('val_augs'))
        ])
    if args.get('test_augs'):
        test_augs = ecvl.SequentialAugmentationContainer([
            ecvl.AugResizeDim(size), ecvl.AugmentationFactory.create(args.get('test_augs'))
        ])

    logger.print_log('Reading dataset')
    dataset_path = dataset.get('path')
    ctypes = [eval(dataset.get('ctype'))]
    if dataset.get('ctype_gt'):
        ctypes.append(eval(dataset.get('ctype_gt')))
    d = ecvl.DLDataset(dataset_path, batch_size, ecvl.DatasetAugmentations([train_augs, val_augs, test_augs]),
                       *ctypes)
    num_classes = len(d.classes_)
    # in_ = eddl.Input([d.n_channels_, size[0], size[1]])
    # out = model(in_, num_classes)  # out is already softmaxed in classific models
    # net = eddl.Model([in_], [out])

    if train:
        eddl.build(
            net,
            eddl.adam(args.get('lr')),
            [bindings.losses_binding.get(args.get('loss'))],
            [bindings.metrics_binding.get(args.get('metric'))],
            eddl.CS_GPU([1], mem='low_mem') if args.get('gpu') else eddl.CS_CPU()
        )
    else:  # inference
        eddl.build(
            net,
            o=eddl.adam(args.get('lr')),
            cs=eddl.CS_GPU([1], mem='low_mem') if args.get('gpu') else eddl.CS_CPU(),
            init_weights=False
        )
    net.resize(batch_size)  # resize manually since we don't use "fit"
    eddl.summary(net)

    # Create tensor for images and labels
    images = Tensor([batch_size, d.n_channels_, size[0], size[1]])
    labels = Tensor([batch_size, num_classes])

    logger.print_log(f'Starting {args.get("mode")}')
    if train:
        num_samples_train = len(d.GetSplit(ecvl.SplitType.training))
        num_batches_train = num_samples_train // batch_size
        num_samples_val = len(d.GetSplit(ecvl.SplitType.validation))
        num_batches_val = num_samples_val // batch_size

        indices = list(range(batch_size))

        for e in range(epochs):
            eddl.reset_loss(net)
            d.SetSplit(ecvl.SplitType.training)
            s = d.GetSplit()
            random.shuffle(s)
            d.split_.training_ = s
            d.ResetCurrentBatch()
            for i in range(num_batches_train):
                d.LoadBatch(images, labels)
                images.div_(255.0)
                eddl.train_batch(net, [images], [labels], indices)

                losses = eddl.get_losses(net)
                metrics = eddl.get_metrics(net)

                logger.print_log(f'Train Epoch: {e + 1}/{epochs} [{i + 1}/{num_batches_train}]'
                                 f'{net.losses[0].name}={losses[0]:.3f} - {net.metrics[0].name}={metrics[0]:.3f}')

            eddl.save_net_to_onnx_file(net, opjoin(ckpts_dir, f'{weight.get("id")}.onnx'))
            logger.print_log('Weights saved')

            if len(d.split_.validation_) > 0:
                logger.print_log(f'Validation {e}/{epochs}')

                d.SetSplit(ecvl.SplitType.validation)
                d.ResetCurrentBatch()

                for i in range(num_batches_val):
                    d.LoadBatch(images, labels)
                    images.div_(255.0)
                    eddl.eval_batch(net, [images], [labels], indices)

                    losses = eddl.get_losses(net)
                    metrics = eddl.get_metrics(net)
                    logger.print_log(
                        f'Validation Epoch: {e + 1}/{epochs} [{i + 1}/{num_batches_train}] {net.lout[0].name}'
                        f'({net.losses[0].name}={losses[0]:1.3f},'
                        f'{net.metrics[0].name}={metrics[0]:1.3f})')
    else:
        d.SetSplit(ecvl.SplitType.test)
        num_samples_test = len(d.GetSplit())
        num_batches_test = num_samples_test // batch_size
        preds = np.empty((0, num_classes), np.float64)

        for b in range(num_batches_test):
            d.LoadBatch(images)
            images.div_(255.0)
            eddl.forward(net, [images])

            logger.print_log(f'Inference Batch {b + 1}/{num_batches_test}')
            # SaveSave network predictions
            for i in range(batch_size):
                pred = np.array(eddl.getOutput(eddl.getOut(net)[0]).select([str(i)]), copy=False)
                # gt = np.argmax(np.array(labels)[indices])
                # pred = np.append(pred, gt).reshape((1, num_classes + 1))
                preds = np.append(preds, pred, axis=0)
                pred_name = d.samples_[d.GetSplit()[b * batch_size + i]].location_
                # print(f'{pred_name};{pred}')
                outputfile.write(f'{pred_name};{pred.tolist()}\n')
        outputfile.close()
    logger.print_log('<done>')
    logger.close()
    del net
    return
