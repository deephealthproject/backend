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
from backend_app import models as dj_models
from deeplearning import bindings
from deeplearning.utils import Logger, dotdict


@shared_task
def classificate(args):
    args = dotdict(args)

    ckpts_dir = opjoin(settings.TRAINING_DIR, 'ckpts')
    outputfile = None
    inference = None
    training = None

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

    # Define augmentations for splits
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

    logger.print_log('Reading dataset')
    dataset = dataset(dataset_path, batch_size, ecvl.DatasetAugmentations([train_augs, val_augs, test_augs]))
    d = dataset.d
    num_classes = dataset.num_classes
    # in_ = eddl.Input([d.n_channels_, size[0], size[1]])
    # out = model(in_, num_classes)  # out is already softmaxed in classific models
    # net = eddl.Model([in_], [out])

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

    # Create tensor for images and labels
    images = Tensor([batch_size, d.n_channels_, size[0], size[1]])
    labels = Tensor([batch_size, num_classes])

    logger.print_log(f'Starting {args.mode}')
    if train:
        num_samples_train = len(d.GetSplit(ecvl.SplitType.training))
        num_batches_train = num_samples_train // batch_size
        num_samples_val = len(d.GetSplit(ecvl.SplitType.validation))
        num_batches_val = num_samples_val // batch_size

        indices = list(range(batch_size))

        for e in range(args.epochs):
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

                logger.print_log(f'Train Epoch: {e + 1}/{args.epochs} [{i + 1}/{num_batches_train}]'
                                 f'{net.losses[0].name}={losses[0]:.3f} - {net.metrics[0].name}={metrics[0]:.3f}')

            eddl.save_net_to_onnx_file(net, opjoin(ckpts_dir, f'{weight_id}.onnx'))
            logger.print_log('Weights saved')

            if len(d.split_.validation_) > 0:
                logger.print_log(f'Validation {e}/{args.epochs}')

                d.SetSplit(ecvl.SplitType.validation)
                d.ResetCurrentBatch()

                for i in range(num_batches_val):
                    d.LoadBatch(images, labels)
                    images.div_(255.0)
                    eddl.eval_batch(net, [images], [labels], indices)

                    losses = eddl.get_losses(net)
                    metrics = eddl.get_metrics(net)
                    logger.print_log(
                        f'Validation Epoch: {e + 1}/{args.epochs} [{i + 1}/{num_batches_train}] {net.lout[0].name}'
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
            # Save network predictions
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
