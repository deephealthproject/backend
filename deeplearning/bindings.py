from deeplearning import datasets
from deeplearning.models import lenet, vgg, segnet

losses_binding = {
    'CrossEntropy': 'cross_entropy',
    'SoftCrossEntropy': 'soft_cross_entropy',
    'MSE': 'mean_squared_error',
}
metrics_binding = {
    'CategoricalAccuracy': 'categorical_accuracy',
    'MSE': 'mean_squared_error',
    'MAE': 'mean_absolute_error',
    'MRE': 'mean_relative_error',
    # 'SUM': '',
}
dataset_binding = {
    1: datasets.MNIST,
    5: datasets.ISICSEG,
    55: datasets.ISICCLAS,
}

models_binding = {
    1: lenet.LeNet,
    2: vgg.VGG16,
    4: segnet.SegNet,
}
