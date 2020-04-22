import pyecvl._core.ecvl as ecvl


class MNIST:
    def __init__(self, path, batch_size,
                 size=[28, 28],
                 num_classes=10,
                 ctype=ecvl.ColorType.GRAY):
        self.d = ecvl.DLDataset(path, batch_size, size, ctype)
        self.num_classes = num_classes


class ISICSEG:
    def __init__(self, path, batch_size,
                 size=[192, 192],
                 num_classes=1,
                 ctype=ecvl.ColorType.BGR,
                 ctype_gt=ecvl.ColorType.GRAY):
        self.d = ecvl.DLDataset(path, batch_size, size, ctype, ctype_gt)
        self.num_classes = num_classes
