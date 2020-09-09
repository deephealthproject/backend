import logging
import sys
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import os
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

import pyecvl.ecvl as ecvl
import random
from celery import shared_task
from os.path import join as opjoin

from backend import settings
from backend_app import models as dj_models
from deeplearning import bindings
from deeplearning.utils import dotdict


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
    # ctype = ecvl.ColorType.GRAY
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
    out_ = eddl.Sigmoid(model(in_, num_classes))
    net = eddl.Model([in_], [out_])
    if train:
        logfile = open(Path(training.logfile), 'w')
    else:
        logfile = open(Path(inference.logfile), 'w')
        outputfile = open(inference.outputfile, 'w')
        output_dir = inference.outputfile[:-4] + '.d'
        os.makedirs(output_dir, exist_ok=True)

    with redirect_stdout(logfile):
        # Save args to file
        print('args: ' + json.dumps(args, indent=2, sort_keys=True), flush=True)
        logging.info('args: ' + json.dumps(args, indent=2, sort_keys=True))

        eddl.build(
            net,
            eddl.adam(args.lr),
            [bindings.losses_binding.get(args.loss)],
            [bindings.metrics_binding.get(args.metric)],
            eddl.CS_GPU([1], mem='low_mem') if args.gpu else eddl.CS_CPU()
        )
        eddl.summary(net)

        if pretrained and os.path.exists(pretrained):
            eddl.load(net, pretrained)
            logging.info('Weights loaded')

        images = Tensor([batch_size, d.n_channels_, size[0], size[1]])
        if train:
            gts = Tensor([batch_size, d.n_channels_gt_, size[0], size[1]])

        # TODO create gts also in test if they exist

        logging.info(f'Starting {args.mode}')
        print(f'Starting {args.mode}', flush=True)
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

                if len(d.split_.validation_) > 0:
                    logging.info(f'Validation {e + 1}/{args.epochs}')
                    print(f'Validation {e + 1}/{args.epochs}', flush=True)
                    d.SetSplit(ecvl.SplitType.validation)
                    evaluator.ResetEval()
                    for j in range(num_batches_val):
                        logging.info(f'Val Epoch: {e + 1}/{args.epochs}  [{j + 1}/{num_batches_val}]')
                        print(f'Val Epoch: {e + 1}/{args.epochs}  [{j + 1}/{num_batches_val}] ', end='', flush=True)
                        d.LoadBatch(images, gts)
                        images.div_(255.0)
                        gts.div_(255.0)
                        eddl.forward(net, [images])
                        output = eddl.getOutput(out_)
                        for k in range(batch_size):
                            img_ = output.select([str(k)])
                            gts_ = gts.select([str(k)])
                            a, b = np.array(img_, copy=False), np.array(gts_, copy=False)
                            iou = evaluator.BinaryIoU(a, b)
                            logging.info('- IoU: %.6g ' % iou)
                            print('- IoU: %.6g ' % iou, flush=True)

                    last_miou = evaluator.MIoU()
                    print(f'Val Epoch: {e + 1}/{args.epochs} - MIoU: {last_miou:.6f}', flush=True)
                    logging.info(f'Val Epoch: {e + 1}/{args.epochs} - MIoU: {last_miou:.6f}')

                    if last_miou > miou:
                        miou = last_miou
                        eddl.save(net, opjoin(ckpts_dir, f'{weight_id}.bin'))
                        logging.info('Weights saved')
                        print('Weights saved')
                else:
                    eddl.save(net, opjoin(ckpts_dir, f'{weight_id}.bin'))
                    print('Weights saved')
        else:
            d.SetSplit(ecvl.SplitType.test)
            num_samples_test = len(d.GetSplit())
            num_batches_test = num_samples_test // batch_size
            for j in range(num_batches_test):
                print(f'Infer Batch {j + 1}/{num_batches_test}', flush=True)
                logging.info(f'Infer Batch {j + 1}/{num_batches_test}')
                d.LoadBatch(images)
                images.div_(255.0)
                eddl.forward(net, [images])
                preds = eddl.getOutput(out_)

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
            print('Inference completed', flush=True)
        print('<done>', flush=True)
    logfile.close()
    del net
    del out_
    del in_
    return
