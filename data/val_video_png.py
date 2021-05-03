import os
from data import srdata

class val_video_png(srdata.SRData):
    def __init__(self, args, name='val_video_png', train=False, benchmark=False):
        super(val_video_png, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(val_video_png, self)._set_filesystem(data_dir)
        self.dir_lr = os.path.join(self.apath, self.args.data_val_compress + '/' + str(self.args.video_index + 1).zfill(3))
        self.ext = ('', '.png')

