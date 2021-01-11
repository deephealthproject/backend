from deeplearning import datasets

losses_binding = {
    'CrossEntropy': 'cross_entropy',
    'Cross Entropy': 'cross_entropy',
    'SoftCrossEntropy': 'soft_cross_entropy',
    'MSE': 'mean_squared_error',
    'BCE': 'binary_cross_entropy',
    'Dice': 'dice',
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
