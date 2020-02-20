import logging
import sys
from contextlib import redirect_stdout

import os
import numpy as np
# import pyecvl._core.ecvl as ecvl
import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT
import random
from celery import shared_task

from backend_app import models as dj_models
from deeplearning import bindings
from deeplearning.utils import dotdict
from backend import settings


@shared_task
def training(args):
    logs_dir = os.path.join(settings.TRAINING_DIR, 'logs/')
    preds_dir = os.path.join(settings.TRAINING_DIR, 'predictions/')
    ckpts_dir = os.path.join(settings.TRAINING_DIR, 'ckpts/')
    os.makedirs(os.path.dirname(logs_dir), exist_ok=True)
    os.makedirs(os.path.dirname(preds_dir), exist_ok=True)

    args = dotdict(args)
    weight_id = args.weight_id
    weight = dj_models.ModelWeights.objects.get(id=weight_id)
    pretrained = None
    if weight.pretrained_on:
        pretrained = weight.pretrained_on.location
    save_stdout = sys.stdout
    size = [28, 28]  # todo: create a property for size
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

    with open(f'{logs_dir}/{weight_id}.log', 'w') as f:
        with redirect_stdout(f):
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
                        f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})')

                    logging.info(
                        f'Batch {i + 1}/{num_batches} {net.lout[0].name}({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                        f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})')

            eddl.save(net, f'{ckpts_dir}/{weight_id}.bin', 'bin')
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
                    f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})')
                logging.info(
                    f'Evaluation {i + 1}/{num_batches} {net.lout[0].name}({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                    f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})')
            print('<done>')
    del net
    del out
    del in_
    return 0


@shared_task
def inference(args):
    logs_dir = os.path.join(settings.INFERENCE_DIR, 'logs/')
    preds_dir = os.path.join(settings.INFERENCE_DIR, 'predictions/')
    os.makedirs(os.path.dirname(logs_dir), exist_ok=True)
    os.makedirs(os.path.dirname(preds_dir), exist_ok=True)

    args = dotdict(args)
    weight_id = args.weight_id
    weight = dj_models.ModelWeights.objects.get(id=weight_id)
    pretrained = weight.location
    save_stdout = sys.stdout
    size = [28, 28]  # todo: create a property for size
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
    layer = in_
    out = model(layer, num_classes)  # out is already softmaxed
    net = eddl.Model([in_], [out])

    with open(f'{logs_dir}/{weight_id}.log', 'w') as f:
        with redirect_stdout(f):
            eddl.build(
                net,
                eddl.sgd(),
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

            images = eddlT.create([args.batch_size, d.n_channels_, size[0], size[1]])
            labels = eddlT.create([args.batch_size, len(d.classes_)])
            indices = list(range(args.batch_size))

            logging.info('Starting inference')
            d.SetSplit('test')
            num_samples = len(d.GetSplit())
            num_batches = num_samples // args.batch_size
            preds = np.empty((0, num_classes + 1), np.float64)

            d.ResetCurrentBatch()

            # start_index = d.current_batch_[d.current_split_] * d.batch_size_
            # samples = d.GetSplit()[start_index:start_index + d.batch_size_]
            # names = [d.samples_[s].location_ for s in samples]

            for b in range(num_batches):
                d.LoadBatch(images, labels)
                images.div_(255.0)
                eddl.eval_batch(net, [images], [labels], indices)
                total_loss = net.fiterr[0]
                total_metric = net.fiterr[1]
                print(
                    f'Evaluation {b + 1}/{num_batches} {net.lout[0].name}({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                    f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})')
                logging.info(
                    f'Evaluation {b + 1}/{num_batches} {net.lout[0].name}({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                    f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})')
                # Save network predictions
                for i in range(args.batch_size):
                    pred = np.array(eddlT.select(eddl.getTensor(out), i), dtype=np.float64)
                    gt = np.argmax(np.array(labels)[indices])
                    pred = np.append(pred, gt).reshape((1, num_classes + 1))
                    preds = np.append(preds, pred, axis=0)
            print('<done>')
            np.save(f'{preds_dir}/{weight_id}.npy', preds.astype(np.float64))
    del net
    del out
    del in_
    return 0
