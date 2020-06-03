import pyeddl.eddl as eddl


def SegNetBN(x, num_classes):
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same")))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 128, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 128, [3, 3], [1, 1], "same")))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.MaxPool(x, [2, 2], [2, 2])

    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 128, [3, 3], [1, 1], "same")))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 128, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same")))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same")))
    x = eddl.Conv(x, num_classes, [3, 3], [1, 1], "same")

    return x
