from deeplearning import datasets
from deeplearning.models import lenet, vgg, segnet, segnetBN

losses_binding = {
    'CrossEntropy': 'cross_entropy',
    'Cross Entropy': 'cross_entropy',
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
    2: datasets.ISICSEG,
    3: datasets.ISICCLAS,
    4: datasets.Pneumothorax,
}

models_binding = {
    1: lenet.LeNet,
    2: vgg.VGG16,
    4: segnet.SegNet,
    5: segnetBN.SegNetBN,
}
