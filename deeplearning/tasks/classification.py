import json
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


@shared_task
def classificate(args):
    ckpts_dir = opjoin(settings.TRAINING_DIR, 'ckpts')
    outputfile = None

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
    d = ecvl.DLDataset(dataset_path, batch_size, dataset_augs, *ctypes)
    num_classes = len(d.classes_)

    net = eddl.import_net_from_onnx_file(net.get('location'), input_shape=[d.n_channels_, *size])

    if train:
        if args.get('remove_layer') and layer_to_remove:
            l_ = eddl.getLayer(net, layer_to_remove)
            if num_classes != l_.output.shape[1]:
                # Classification layer must be replaced
                net_layer_names = [l.name for l in net.layers]
                layer_to_remove_index = net_layer_names.index(layer_to_remove)
                # Remove all layers from the end to "layer_to_remove"
                for b in range(len(net_layer_names) - 1, layer_to_remove_index - 1, -1):
                    eddl.removeLayer(net, net_layer_names[b])
                top = eddl.getLayer(net, net_layer_names[layer_to_remove_index - 1])
                out = eddl.Softmax(eddl.Dense(top, num_classes, use_bias=True, name=FINAL_LAYER))

                # Retrieve the name of the input layer
                l_input = None
                for l in net.layers:
                    if isinstance(l, eddl_core.LInput):
                        l_input = eddl.getLayer(net, l.name)
                        break
                assert l_input is not None
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
            eddl.adam(args.get('lr')),
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
        eddl.set_mode(net, 1)
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
            d.ResetAllBatches()

            for b in range(num_batches_train):
                d.LoadBatch(images, labels)
                eddl.train_batch(net, [images], [labels], indices)

                losses = eddl.get_losses(net)
                metrics = eddl.get_metrics(net)

                logger.print_log(f'Train - epoch [{e + 1}/{epochs}] - batch [{b + 1}/{num_batches_train}]'
                                 f' - loss={losses[0]:.3f} - metric={metrics[0]:.3f}')

            eddl.save_net_to_onnx_file(net, opjoin(ckpts_dir, f'{weight.get("id")}.onnx'))
            logger.print_log('Weights saved')

            if len(d.split_.validation_) > 0:
                eddl.reset_loss(net)
                logger.print_log(f'Validation {e}/{epochs}')
                d.SetSplit(ecvl.SplitType.validation)

                for b in range(num_batches_val):
                    d.LoadBatch(images, labels)
                    eddl.eval_batch(net, [images], [labels], indices)
                    losses = eddl.get_losses(net)
                    metrics = eddl.get_metrics(net)
                    logger.print_log(
                        f'Validation - epoch [{e + 1}/{epochs}] - batch [{b + 1}/{num_batches_val}]'
                        f' - loss={losses[0]:.3f} - metric={metrics[0]:.3f}')
    else:
        eddl.set_mode(net, 0)
        d.SetSplit(ecvl.SplitType.test)
        num_samples_test = len(d.GetSplit())
        num_batches_test = num_samples_test // batch_size
        d.ResetAllBatches()

        # Check if this dataset has labels or not
        split_samples_np = np.take(d.samples_, d.GetSplit())
        split_samples_have_labels = all(s.label_ is not None or s.label_path_ is not None for s in split_samples_np)

        preds = np.empty((0, num_classes), np.float64)
        values = np.zeros(num_batches_test)
        out_layer = eddl.getOut(net)[0]
        metric_fn = eddl.getMetric(args.get('metric'))

        if split_samples_have_labels:
            # The dataset has labels for the images, we can show the metric over the predictions
            for b in range(num_batches_test):
                d.LoadBatch(images, labels)
                # index = 0
                # tmp = images.select([str(index)])
                # tmp.mult_(255)
                # tmp.save(f"images_test/{b}_{index}.jpg")
                eddl.forward(net, [images])

                output = eddl.getOutput(out_layer)
                value = metric_fn.value(labels, output)
                values[b] = value

                logger.print_log(f'Inference - batch [{b + 1}/{num_batches_test}] -'
                                 f' metric={np.mean(values[:b + 1] / batch_size):.3f}')

                # Save network predictions
                for bs in range(batch_size):
                    pred = np.array(output.select([str(bs)]), copy=False)
                    # gt = np.argmax(np.array(labels)[indices])
                    # pred = np.append(pred, gt).reshape((1, num_classes + 1))
                    preds = np.append(preds, pred, axis=0)
                    pred_name = d.samples_[d.GetSplit()[b * batch_size + bs]].location_
                    # print(f'{pred_name};{pred}')
                    outputfile.write(f'{pred_name};{pred.tolist()}\n')
            logger.print_log(f'Inference - metric={np.mean(values / batch_size):.3f}')
        else:
            for b in range(num_batches_test):
                d.LoadBatch(images)
                eddl.forward(net, [images])

                logger.print_log(f'Inference - batch [{b + 1}/{num_batches_test}]')
                # Save network predictions
                output = eddl.getOutput(out_layer)
                for bs in range(batch_size):
                    pred = np.array(output.select([str(bs)]), copy=False)
                    # gt = np.argmax(np.array(labels)[indices])
                    # pred = np.append(pred, gt).reshape((1, num_classes + 1))
                    preds = np.append(preds, pred, axis=0)
                    pred_name = d.samples_[d.GetSplit()[b * batch_size + bs]].location_
                    # print(f'{pred_name};{pred}')
                    outputfile.write(f'{pred_name};{pred.tolist()}\n')
        outputfile.close()
    logger.print_log('<done>')
    logger.close()
    del net
    return
