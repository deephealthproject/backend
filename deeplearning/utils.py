import ctypes
import itertools

from backend import settings
from backend_app import models


# from django.forms.models import model_to_dict


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


dotdict = DotDict


def cuda_is_available():
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError("could not load any of: " + ' '.join(libnames))
    CUDA_SUCCESS = 0
    nGpus = ctypes.c_int()
    result = ctypes.c_int()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        print("cuInit failed with error code %d: %s" % (result, error_str.value.decode()))
        return False
    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        print("cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
        return False
    return True


def createConfig(task, hyperparams, mode: str):
    assert mode in ['training', 'inference']
    try:
        config = {
            'weight': model_to_dict(task.modelweights_id, fields=['id']),
            'input_h': int(hyperparams.get('Input height')),
            'input_w': int(hyperparams.get('Input width')),
            'metric': str(hyperparams.get('Metric')),
            'batch_size': int(hyperparams.get('Batch size')),

            'mode': mode,
            'log_interval': 50,
            'save_model': True,
            'gpu': settings.env('EDDL_WITH_CUDA'),
        }
    except TypeError as e:
        print(f"Type error: {e}")
        return False

    if mode == 'training':
        if task.modelweights_id.pretrained_on:  # Finetuning on an existing weight
            net = model_to_dict(task.modelweights_id.pretrained_on, fields=['id', 'location'])
        else:  # Train from scratch
            net = model_to_dict(task.modelweights_id.model_id, fields=['id', 'location'])

        config.update({
            'net': net,
            'dataset': model_to_dict(task.modelweights_id.dataset_id, fields=['id', 'path', 'ctype', 'ctype_gt']),
            'task': model_to_dict(task, fields=['id', 'logfile']),
            'lr': abs(float(hyperparams.get('Learning rate'))),
            'epochs': int(hyperparams.get('Epochs')),
            'loss': str(hyperparams.get('Loss function')),
            'train_augs': (hyperparams.get('Training augmentations')),
            'val_augs': (hyperparams.get('Validation augmentations')),
            'split': 'training',
        })
    else:
        config.update({
            'net': model_to_dict(task.modelweights_id, fields=['id', 'location']),
            'dataset': model_to_dict(task.dataset_id, fields=['id', 'path', 'ctype', 'ctype_gt']),
            'task': model_to_dict(task, fields=['id', 'logfile', 'outputfile']),
            'lr': 0.1,
            'test_augs': (hyperparams.get('Test augmentations')),
            'split': 'test',
        })

    return config


def nn_settings(training, hyperparams, dataset_id=None, mode='training'):
    modelweight = training.modelweights_id
    if dataset_id is None:
        dataset_id = modelweight.dataset_id_id

    try:
        config = {
            'training_id': training.id,
            'weight_id': modelweight.id,
            'model_id': modelweight.model_id_id,
            'dataset_id': dataset_id,
            'pretrained': modelweight.pretrained_on_id,

            'input_h': int(hyperparams.get('Input height')),
            'input_w': int(hyperparams.get('Input width')),

            'lr': abs(float(hyperparams.get('Learning rate'))),
            'epochs': int(hyperparams.get('Epochs')),
            'loss': str(hyperparams.get('Loss function')),
            'metric': str(hyperparams.get('Metric')),
            'batch_size': int(hyperparams.get('Batch size')),
            'test_batch_size': 1,
            'train_augs': (hyperparams.get('Training augmentations')),
            'val_augs': (hyperparams.get('Validation augmentations')),

            'mode': mode,
            'split': 'training',
            'log_interval': 50,
            'save_model': True,
            'gpu': settings.env('EDDL_WITH_CUDA'),
        }
    except TypeError as e:
        print(f"Type error: {e}")
        return False

    return config


def inference_settings(inference_id, hyperparams, dataset_id=None):
    modelweight = models.Inference.objects.get(id=inference_id).modelweights_id

    if dataset_id is None:
        dataset_id = modelweight.dataset_id_id

    try:
        config = {
            'inference_id': inference_id,
            'weight_id': modelweight.id,
            'model_id': modelweight.model_id_id,
            'dataset_id': dataset_id,
            'pretrained': modelweight.pretrained_on_id,

            'input_h': int(hyperparams.get('Input height')),
            'input_w': int(hyperparams.get('Input width')),

            'lr': 0.1,  # not used in inference
            'epochs': int(hyperparams.get('Epochs')),  # not used in inference
            'loss': str(hyperparams.get('Loss function')),  # not used in inference
            'metric': str(hyperparams.get('Metric')),
            'test_batch_size': 1,
            'test_augs': (hyperparams.get('Test augmentations')),

            'mode': 'inference',
            'split': 'test',
            'log_interval': 50,
            'save_model': True,
            'gpu': settings.env('EDDL_WITH_CUDA'),
        }
    except TypeError as e:
        print(f"Type error: {e}")
        return False

    return config


class Logger:
    def __init__(self, filename=None, filemode='w'):
        self.file = None
        if filename:
            self.file = open(filename, filemode)

    def open(self, filename=None, filemode='w'):
        self.file = open(filename, filemode)

    def close(self):
        self.file.close()

    def log(self, message, end='\n'):
        assert self.file is not None
        print(message, end=end, file=self.file, flush=True)

    def print(self, message, end='\n'):
        print(message, end=end, flush=True)

    def print_log(self, message, end='\n'):
        self.print(message, end=end)
        self.log(message, end=end)


def model_to_dict(instance, fields=None, exclude=None):
    """
    Return a dict containing the data in ``instance`` suitable for passing as
    a Form's ``initial`` keyword argument.

    ``fields`` is an optional list of field names. If provided, return only the
    named.

    ``exclude`` is an optional list of field names. If provided, exclude the
    named from the returned dict, even if they are listed in the ``fields``
    argument.
    """
    opts = instance._meta
    data = {}
    for f in itertools.chain(opts.concrete_fields, opts.private_fields, opts.many_to_many):
        if not getattr(f, 'editable', False):
            continue
        if fields is not None and f.name not in fields:
            continue
        if exclude and f.name in exclude:
            continue
        if f.choices is not None:
            data[f.name] = eval(f'instance.get_{f.name}_display()')
        else:
            data[f.name] = f.value_from_object(instance)
    return data
