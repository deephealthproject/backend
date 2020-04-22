import json
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import os
# import pyecvl._core.ecvl as ecvl
import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT
import random
from celery import shared_task
from os.path import join as opjoin

from backend import settings
from backend_app import models as dj_models
from deeplearning import bindings
from deeplearning.utils import dotdict


@shared_task
def training(args):
    ckpts_dir = opjoin(settings.TRAINING_DIR, 'ckpts')
    os.makedirs(os.path.dirname(ckpts_dir), exist_ok=True)

    args = dotdict(args)
    weight_id = args.weight_id
    weight = dj_models.ModelWeights.objects.get(id=weight_id)
    pretrained = None
    if weight.pretrained_on:
        pretrained = weight.pretrained_on.location
    save_stdout = sys.stdout
    size = [args.input_h, args.input_w]  # Height, width
    try:
        model = bindings.models_binding.get(args.model_id)
    except KeyError:
        return -1
    try:
        dataset_path = str(dj_models.Dataset.objects.get(id=args.dataset_id).path)
        dataset = bindings.dataset_binding.get(args.dataset_id)
    except KeyError:
        return -1

    dataset = dataset(dataset_path, args.batch_size, size)
    d = dataset.d
    num_classes = dataset.num_classes
    in_ = eddl.Input([1, size[0], size[1]])
    out = model(in_, num_classes)
    net = eddl.Model([in_], [out])

    logfile = open(Path(weight.logfile), 'w')
    with redirect_stdout(logfile):
        eddl.build(
            net,
            eddl.sgd(args.lr, 0.9),
            [bindings.losses_binding.get(args.loss)],
            [bindings.metrics_binding.get(args.metric)],
            eddl.CS_GPU([1]) if args.gpu else eddl.CS_CPU()
        )
        eddl.summary(net)

        if pretrained and os.path.exists(pretrained):
            eddl.load(net, pretrained)
            logging.info('Weights loaded')

        logging.info('Reading dataset')

        images = eddlT.create([args.batch_size, d.n_channels_, size[0], size[1]])
        labels = eddlT.create([args.batch_size, len(d.classes_)])
        num_samples = len(d.GetSplit())
        num_batches = num_samples // args.batch_size
        indices = list(range(args.batch_size))

        for e in range(args.epochs):
            eddl.reset_loss(net)
            d.SetSplit('training')
            s = d.GetSplit()
            num_samples = len(s)
            random.shuffle(s)
            d.split_.training_ = s
            d.ResetCurrentBatch()
            # total_loss = 0.
            # total_metric = 0.
            for i in range(num_batches):
                d.LoadBatch(images, labels)
                images.div_(255.0)
                tx, ty = [images], [labels]
                eddl.train_batch(net, tx, ty, indices)
                total_loss = net.fiterr[0]
                total_metric = net.fiterr[1]
                print(
                    f'Batch {i + 1}/{num_batches} {net.lout[0].name}({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                    f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})', flush=True)

                logging.info(
                    f'Batch {i + 1}/{num_batches} {net.lout[0].name}({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                    f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})')

        eddl.save(net, f'{ckpts_dir}/{weight_id}.bin')
        logging.info('Weights saved')

        logging.info('Evaluation')
        d.SetSplit('test')
        num_samples = len(d.GetSplit())
        num_batches = num_samples // args.batch_size

        d.ResetCurrentBatch()

        for i in range(num_batches):
            d.LoadBatch(images, labels)
            images.div_(255.0)
            eddl.eval_batch(net, [images], [labels], indices)
            # eddl.evaluate(net, [images], [labels])

            total_loss = net.fiterr[0]
            total_metric = net.fiterr[1]
            print(
                f'Evaluation {i + 1}/{num_batches} {net.lout[0].name}({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})', flush=True)
            logging.info(
                f'Evaluation {i + 1}/{num_batches} {net.lout[0].name}({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})')
        print('<done>')
    logfile.close()
    del net
    del out
    del in_
    return 0


@shared_task
def inference(args):
    args = dotdict(args)
    inference_id = args.inference_id
    inference = dj_models.Inference.objects.get(id=inference_id)
    batch_size = args.test_batch_size
    weight_id = args.weight_id
    weight = dj_models.ModelWeights.objects.get(id=weight_id)
    pretrained = weight.location
    save_stdout = sys.stdout
    size = [args.input_h, args.input_w]  # Height, width
    try:
        model = bindings.models_binding.get(args.model_id)
    except KeyError:
        return -1
    try:
        dataset_path = str(dj_models.Dataset.objects.get(id=args.dataset_id).path)
        dataset = bindings.dataset_binding.get(args.dataset_id)
    except KeyError:
        return -1

    if dataset is None:
        # Binding does not exist. it's a single image dataset
        # Use as dataset "stub" the dataset on which model has been trained
        dataset = bindings.dataset_binding.get(weight.dataset_id.id)

    dataset = dataset(dataset_path, batch_size, size)
    d = dataset.d
    num_classes = dataset.num_classes
    in_ = eddl.Input([1, size[0], size[1]])
    layer = in_
    out = model(layer, num_classes)  # out is already softmaxed
    net = eddl.Model([in_], [out])

    logfile = open(inference.logfile, 'w')
    outputfile = open(inference.outputfile, 'w')
    with redirect_stdout(logfile):
        # Save args to file
        print('args: ' + json.dumps(args, indent=2, sort_keys=True))

        eddl.build(
            net,
            eddl.sgd(0.001),
            [bindings.losses_binding.get(args.loss)],
            [bindings.metrics_binding.get(args.metric)],
        )

        eddl.summary(net)

        if args.gpu:
            eddl.toGPU(net, [1])

        if os.path.exists(pretrained):
            eddl.load(net, pretrained)
            logging.info('Weights loaded')
        else:
            return -1
        logging.info('Reading dataset')

        images = eddlT.create([batch_size, d.n_channels_, size[0], size[1]])
        # labels = eddlT.create([batch_size, 1])
        labels = eddlT.create([batch_size, num_classes])

        logging.info('Starting inference')
        d.SetSplit('test')
        num_samples = len(d.GetSplit())
        num_batches = num_samples // batch_size
        preds = np.empty((0, num_classes), np.float64)

        for b in range(num_batches):
            d.LoadBatch(images, labels)
            images.div_(255.0)
            eddl.forward(net, [images])

            # total_loss = net.fiterr[0]
            # total_metric = net.fiterr[1]
            print(f'Inference {b + 1}/{num_batches}', flush=True)
            logging.info(f'Inference {b + 1}/{num_batches}')

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
        print('<done>', flush=True)
        # np.save(f'{preds_dir}/{weight_id}.npy', preds.astype(np.float64))
    logfile.close()
    outputfile.close()
    del net
    del out
    del in_
    return 0
