import pyecvl._core.ecvl as ecvl


class DATESET:
    def __init__(self, path, batch_size, augs,
                 ctype=ecvl.ColorType.GRAY):
        self.d = ecvl.DLDataset(path, batch_size, augs, ctype)
        self.num_classes = len(self.d.classes_ or [1])


class MNIST:
    def __init__(self, path, batch_size, augs,
                 ctype=ecvl.ColorType.GRAY):
        self.d = ecvl.DLDataset(path, batch_size, augs, ctype)
        self.num_classes = len(self.d.classes_ or [1])


class ISICCLAS:
    def __init__(self, path, batch_size, augs,
                 ctype=ecvl.ColorType.BGR):
        self.d = ecvl.DLDataset(path, batch_size, augs, ctype)
        self.num_classes = len(self.d.classes_ or [1])


class ISICSEG:
    def __init__(self, path, batch_size, augs,
                 ctype=ecvl.ColorType.BGR,
                 ctype_gt=ecvl.ColorType.GRAY):
        self.d = ecvl.DLDataset(path, batch_size, augs, ctype, ctype_gt)
        self.num_classes = len(self.d.classes_ or [1])
