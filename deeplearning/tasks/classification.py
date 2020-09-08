import json
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import os
import pyeddl.eddl as eddl
import pyeddl.eddlT as eddlT
import pyecvl.ecvl as ecvl
import random
from celery import shared_task
from os.path import join as opjoin

from backend import settings
from backend_app import models as dj_models
from deeplearning import bindings
from deeplearning.utils import dotdict


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
    if train:
        training = dj_models.Training.objects.get(id=args.training_id)
        pretrained = None
        if weight.pretrained_on:
            pretrained = weight.pretrained_on.location
    else:
        inference_id = args.inference_id
        inference = dj_models.Inference.objects.get(id=inference_id)
        pretrained = weight.location
    save_stdout = sys.stdout
    size = [args.input_h, args.input_w]  # Height, width
    try:
        model = bindings.models_binding[args.model_id]
    except KeyError:
        raise Exception(f'Model with id: {args.model_id} not found in bindings.py')
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

    logging.info('Reading dataset')
    print('Reading dataset', flush=True)

    dataset = dataset(dataset_path, batch_size, ecvl.DatasetAugmentations([train_augs, val_augs, test_augs]))
    d = dataset.d
    num_classes = dataset.num_classes
    in_ = eddl.Input([d.n_channels_, size[0], size[1]])
    out = model(in_, num_classes)  # out is already softmaxed in classific models
    net = eddl.Model([in_], [out])

    if train:
        logfile = open(Path(training.logfile), 'w')
    else:
        logfile = open(inference.logfile, 'w')
        outputfile = open(inference.outputfile, 'w')
    with redirect_stdout(logfile):
        # Save args to file
        print('args: ' + json.dumps(args, indent=2, sort_keys=True), flush=True)
        logging.info('args: ' + json.dumps(args, indent=2, sort_keys=True))

        eddl.build(
            net,
            eddl.sgd(args.lr, 0.9),
            [bindings.losses_binding.get(args.loss)],
            [bindings.metrics_binding.get(args.metric)],
            eddl.CS_GPU([1], mem='low_mem') if args.gpu else eddl.CS_CPU()
        )
        eddl.summary(net)

        if pretrained and os.path.exists(pretrained):
            eddl.load(net, pretrained)
            logging.info('Weights loaded')

        # Create tensor for images and labels
        images = eddlT.create([batch_size, d.n_channels_, size[0], size[1]])
        labels = eddlT.create([batch_size, num_classes])

        logging.info(f'Starting {args.mode}')
        print(f'Starting {args.mode}', flush=True)
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
                # total_loss = 0.
                # total_metric = 0.
                for i in range(num_batches_train):
                    d.LoadBatch(images, labels)
                    images.div_(255.0)
                    eddl.train_batch(net, [images], [labels], indices)
                    total_loss = net.fiterr[0]
                    total_metric = net.fiterr[1]
                    print(
                        f'Train Epoch: {e + 1}/{args.epochs} [{i + 1}/{num_batches_train}] {net.lout[0].name}'
                        f'({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                        f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})', flush=True)

                    logging.info(
                        f'Train Epoch: {e + 1}/{args.epochs} [{i + 1}/{num_batches_train}] {net.lout[0].name}'
                        f'({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                        f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})')

                eddl.save(net, opjoin(ckpts_dir, f'{weight_id}.bin'))
                logging.info('Weights saved')
                print('Weights saved', flush=True)

                if len(d.split_.validation_) > 0:

                    logging.info(f'Validation {e}/{args.epochs}')
                    print(f'Validation {e}/{args.epochs}', flush=True)

                    d.SetSplit(ecvl.SplitType.validation)
                    d.ResetCurrentBatch()

                    for i in range(num_batches_val):
                        d.LoadBatch(images, labels)
                        images.div_(255.0)
                        eddl.eval_batch(net, [images], [labels], indices)
                        # eddl.evaluate(net, [images], [labels])

                        total_loss = net.fiterr[0]
                        total_metric = net.fiterr[1]
                        print(
                            f'Val Epoch: {e + 1}/{args.epochs}  [{i + 1}/{num_batches_val}] {net.lout[0].name}'
                            f'({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                            f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})', flush=True)
                        logging.info(
                            f'Val Epoch: {e + 1}/{args.epochs}  [{i + 1}/{num_batches_val}] {net.lout[0].name}'
                            f'({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                            f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})')
        else:
            d.SetSplit(ecvl.SplitType.test)
            num_samples_test = len(d.GetSplit())
            num_batches_test = num_samples_test // batch_size
            preds = np.empty((0, num_classes), np.float64)

            for b in range(num_batches_test):
                d.LoadBatch(images)
                images.div_(255.0)
                eddl.forward(net, [images])

                print(f'Infer Batch {b + 1}/{num_batches_test}', flush=True)
                logging.info(f'Infer Batch {b + 1}/{num_batches_test}')

                # print(
                #     f'Evaluation {b + 1}/{num_batches} {net.lout[0].name}({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                #     f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})')
                # logging.info(
                #     f'Evaluation {b + 1}/{num_batches} {net.lout[0].name}({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                #     f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})')
                # Save network predictions
                for i in range(batch_size):
                    pred = np.array(eddlT.select(eddl.getTensor(out), i), copy=False)
                    # gt = np.argmax(np.array(labels)[indices])
                    # pred = np.append(pred, gt).reshape((1, num_classes + 1))
                    preds = np.append(preds, pred, axis=0)
                    pred_name = d.samples_[d.GetSplit()[b * batch_size + i]].location_
                    # print(f'{pred_name};{pred}')
                    outputfile.write(f'{pred_name};{pred.tolist()}\n')
            outputfile.close()
        print('<done>')
    logfile.close()
    del net
    del out
    del in_
    return
