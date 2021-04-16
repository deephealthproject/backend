import itertools
from typing import Dict, Union

from backend import settings

FINAL_LAYER = 'final_layer'


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


dotdict = DotDict


def createConfig(task, hyperparams: dict, mode: str) -> Union[bool, Dict]:
    """
    Create the configuration dictionary for launching a training/inference job

    @param task: Training or Inference object
    @param hyperparams: dict containing the hyperparameters for training/inference
    @param mode: String that can be 'training' or 'inference'
    @return: A dictionary with information for launch the trainig/inference job
    """
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

    dataset_fields = ['id', 'path', 'ctype', 'ctype_gt', 'classes']
    if mode == 'training':
        # Finetuning on an existing weight or train from scratch using that weight

        # If last_layers are different between current onnx and parent -> you will remove 'layer_to_remove'
        remove_layer = task.modelweights_id.layer_to_remove != task.modelweights_id.pretrained_on.layer_to_remove

        config.update({
            'net': model_to_dict(task.modelweights_id.pretrained_on, fields=['id', 'location', 'layer_to_remove']),
            'dataset': model_to_dict(task.modelweights_id.dataset_id, fields=dataset_fields),
            'task': model_to_dict(task, fields=['id', 'logfile']),
            'lr': abs(float(hyperparams.get('Learning rate'))),
            'epochs': int(hyperparams.get('Epochs')),
            'loss': str(hyperparams.get('Loss function')),
            'train_augs': (hyperparams.get('Training augmentations')),
            'val_augs': (hyperparams.get('Validation augmentations')),
            'split': 'training',
            'remove_layer': remove_layer
        })
    else:
        # Inference
        config.update({
            'net': model_to_dict(task.modelweights_id, fields=['id', 'location']),
            'dataset': model_to_dict(task.dataset_id, fields=dataset_fields),
            'task': model_to_dict(task, fields=['id', 'logfile', 'outputfile']),
            'lr': 0.1,
            'test_augs': (hyperparams.get('Test augmentations')),
            'split': 'test',
        })

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
