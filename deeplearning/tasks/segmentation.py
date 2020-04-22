import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import os
import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT
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
def training(args):
    ckpts_dir = opjoin(settings.TRAINING_DIR, 'ckpts/')
    os.makedirs(os.path.dirname(ckpts_dir), exist_ok=True)

    args = dotdict(args)
    weight_id = args.weight_id
    weight = dj_models.ModelWeights.objects.get(id=weight_id)
    pretrained = None
    if weight.pretrained_on:
        pretrained = weight.pretrained_on.location
    save_stdout = sys.stdout
    size = [args.input_h, args.input_w]  # Height, width
    # ctype = ecvl.ColorType.GRAY
    try:
        model = bindings.models_binding.get(args.model_id)
    except KeyError:
        return 1
    try:
        dataset_path = str(dj_models.Dataset.objects.get(id=args.dataset_id).path)
        dataset = bindings.dataset_binding.get(args.dataset_id)
    except KeyError:
        return 1
    except:
        return 1

    # dataset = dataset(dataset_path, args.batch_size, args.split)
    dataset = dataset(dataset_path, args.batch_size, size)
    d = dataset.d
    num_classes = dataset.num_classes
    in_ = eddl.Input([3, size[0], size[1]])
    out = model(in_, num_classes)
    out_sigm = eddl.Sigmoid(out)
    net = eddl.Model([in_], [out_sigm])

    logfile = open(Path(weight.logfile), 'w')
    with redirect_stdout(logfile):

        eddl.build(
            net,
            # eddl.sgd(args.lr, 0.9),
            eddl.adam(args.lr),
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
        gts = eddlT.create([args.batch_size, 1, size[0], size[1]])
        num_samples = len(d.GetSplit())
        num_batches = num_samples // args.batch_size
        indices = list(range(args.batch_size))

        d.SetSplit('validation')
        num_samples_validation = len(d.GetSplit())
        num_batches_validation = num_samples_validation // args.batch_size
        evaluator = Evaluator()
        miou = -1
        for e in range(args.epochs):
            d.SetSplit('training')
            eddl.reset_loss(net)
            s = d.GetSplit()
            random.shuffle(s)
            d.split_.training_ = s
            d.ResetAllBatches()
            for i in range(num_batches):
                d.LoadBatch(images, gts)
                images.div_(255.0)
                gts.div_(255.0)
                eddl.train_batch(net, [images], [gts], indices)
                total_loss = net.fiterr[0]
                total_metric = net.fiterr[1]
                print(
                    f'Batch {i + 1}/{num_batches} {net.lout[0].name}({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                    f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})', flush=True)

                logging.info(
                    f'Batch {i + 1}/{num_batches} {net.lout[0].name}({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
                    f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})')

            logging.info('Evaluation')
            d.SetSplit('validation')
            evaluator.ResetEval()
            for j in range(num_batches_validation):
                print('Validation - Epoch %d/%d (batch %d/%d) ' %
                      (e + 1, args.epochs, j + 1, num_batches_validation),
                      end='', flush=True)
                d.LoadBatch(images, gts)
                images.div_(255.0)
                gts.div_(255.0)
                eddl.forward(net, [images])
                output = eddl.getTensor(out_sigm)
                for k in range(args.batch_size):
                    img_ = eddlT.select(output, k)
                    gts_ = eddlT.select(gts, k)
                    a, b = np.array(img_, copy=False), np.array(gts_, copy=False)
                    iou = evaluator.BinaryIoU(a, b)
                    print('- IoU: %.6g ' % iou, flush=True)

            last_miou = evaluator.MIoU()
            print(f'Validation MIoU: {last_miou:.6f}', flush=True)
            logging.info(f'Validation MIoU: {last_miou:.6f}')

            if last_miou > miou:
                miou = last_miou
                eddl.save(net, f'{ckpts_dir}/{weight_id}.bin', 'bin')
                logging.info('Weights saved')

        # d.SetSplit('test')
        # num_samples = len(d.GetSplit())
        # num_batches = num_samples // args.batch_size
        #
        # d.ResetCurrentBatch()
        #
        # for i in range(num_batches):
        #     d.LoadBatch(images, gts)
        #     images.div_(255.0)
        #     eddl.eval_batch(net, [images], [gts], indices)
        #     # eddl.evaluate(net, [images], [labels])
        #
        #     total_loss = net.fiterr[0]
        #     total_metric = net.fiterr[1]
        #     print(
        #         f'Evaluation {i + 1}/{num_batches} {net.lout[0].name}({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
        #         f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})')
        #     logging.info(
        #         f'Evaluation {i + 1}/{num_batches} {net.lout[0].name}({net.losses[0].name}={total_loss / net.inferenced_samples:1.3f},'
        #         f'{net.metrics[0].name}={total_metric / net.inferenced_samples:1.3f})')
        # print('<done>')
    logfile.close()
    del net
    del out
    del in_
    return 0


@shared_task
def inference(args):
    args = dotdict(args)
    batch_size = args.test_batch_size

    inference_id = args.inference_id
    inference = dj_models.Inference.objects.get(id=inference_id)
    weight_id = args.weight_id
    weight = dj_models.ModelWeights.objects.get(id=weight_id)
    pretrained = weight.location
    save_stdout = sys.stdout
    size = [args.input_h, args.input_w]  # Height, width
    # ctype = ecvl.ColorType.GRAY
    try:
        model = bindings.models_binding.get(args.model_id)
    except KeyError:
        return 1
    try:
        dataset_path = str(dj_models.Dataset.objects.get(id=args.dataset_id).path)
        dataset = bindings.dataset_binding.get(args.dataset_id)
    except KeyError:
        return 1

    logging.info('Reading dataset')
    dataset = dataset(dataset_path, batch_size, size)
    d = dataset.d
    num_classes = dataset.num_classes
    in_ = eddl.Input([d.n_channels_, size[0], size[1]])
    out_ = eddl.Sigmoid(model(in_, num_classes))
    net = eddl.Model([in_], [out_])

    logfile = open(Path(inference.logfile), 'w')
    outputfile = open(inference.outputfile, 'w')
    output_dir = inference.outputfile[:-4] + '.d'
    os.makedirs(output_dir, exist_ok=True)

    with redirect_stdout(logfile):
        eddl.build(
            net,
            # eddl.sgd(args.lr, 0.9),
            eddl.adam(),
            [bindings.losses_binding.get(args.loss)],
            [bindings.metrics_binding.get(args.metric)],
        )
        eddl.summary(net)

        if args.gpu:
            eddl.toGPU(net, [1], "low_mem")

        if os.path.exists(pretrained):
            eddl.load(net, pretrained)
            logging.info('Weights loaded')
        else:
            return -1

        images = eddlT.create([batch_size, d.n_channels_, size[0], size[1]])
        gts = eddlT.create([batch_size, 1, size[0], size[1]])
        # indices = list(range(batch_size))

        evaluator = Evaluator()
        d.SetSplit('test')
        num_samples = len(d.GetSplit())
        num_batches = num_samples // batch_size

        logging.info('test')
        evaluator.ResetEval()
        for j in range(num_batches):
            d.LoadBatch(images, gts)
            images.div_(255.0)
            gts.div_(255.0)
            # eddl.forward(net, [images])
            indices = np.arange(0, batch_size).tolist()
            eddl.eval_batch(net, [images], [gts], indices)
            preds = eddl.getTensor(out_)
            for k in range(batch_size):
                pred = eddlT.select(preds, k)
                gt = eddlT.select(gts, k)

                pred_np, gt = np.array(pred, copy=False), np.array(gt, copy=False)
                iou = evaluator.BinaryIoU(pred_np, gt)
                print(f'Inference {batch_size * j + k + 1}/{num_batches * batch_size} IoU: {iou:.6f}', flush=True)
                logging.info(f'Inference {batch_size * j + k + 1}/{num_batches * batch_size} IoU: {iou:.6f}')
                pred_np[pred_np >= 0.5] = 255
                pred_np[pred_np < 0.5] = 0

                orig_image_path = d.samples_[d.GetSplit()[j * batch_size + k]].location_
                orig_image_name = Path(orig_image_path).name.split('.')
                orig_image_name = orig_image_name[0] + '.png'

                # TODO PyECVL need to be updated with latest bindings
                # img_ecvl = ecvl.TensorToImage(img)
                # img_ecvl.colortype_ = ecvl.ColorType.GRAY
                # img_ecvl.channels_ = "xyc"
                # ecvl.ImWrite(opjoin(output_dir, orig_image_name), img_ecvl)

                # TODO resize of the gts
                # original_image = ecvl.ImRead(orig_image_path)
                # original_dims = original_image.dims_[:-1]

                eddlT.save(pred, opjoin(output_dir, orig_image_name))
                outputfile.write(opjoin(output_dir, orig_image_name) + '\n')

        print(f'Validation MIoU: {evaluator.MIoU():.6f}', flush=True)
        logging.info(f'Validation MIoU: {evaluator.MIoU():.6f}')
        print('<done>')
    logfile.close()
    outputfile.close()
    del net
    del out_
    del in_
    return 0
