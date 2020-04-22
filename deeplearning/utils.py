from backend_app import models


# class dotdict(dict):
#     """dot.notation access to dictionary attributes"""
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__

class dotdict(dict):
    def __getattr__(self, val):
        return self[val]


def nn_settings(modelweight, hyperparams, dataset_id=None, mode='training'):
    if dataset_id is None:
        dataset_id = modelweight.dataset_id_id

    try:
        config = {
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

            'mode': mode,
            'split': 'training',
            'log_interval': 50,
            'save_model': True,
            'gpu': True,
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

            # 'lr': float(hyperparams.get('Learning rate')),
            'epochs': int(hyperparams.get('Epochs')),
            'loss': str(hyperparams.get('Loss function')),
            'metric': str(hyperparams.get('Metric')),
            'test_batch_size': 1,

            'mode': 'inference',
            'split': 'test',
            'log_interval': 50,
            'save_model': True,
            'gpu': True,
        }
    except TypeError as e:
        print(f"Type error: {e}")
        return False

    return config
