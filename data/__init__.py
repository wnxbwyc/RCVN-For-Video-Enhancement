from importlib import import_module
from torch.utils.data import DataLoader


class Data:
    def __init__(self, args):
        module_test = import_module('data.val_video_png')
        testset = getattr(module_test, 'val_video_png')(args, train=False)
        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            num_workers=args.n_threads,
            shuffle=False,
            pin_memory=not args.cpu
        )
