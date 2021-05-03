import os
import glob
import numpy as np
import imageio
import torch.utils.data as data
import torch

def np2Tensor2(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)
        return tensor
    return [_np2Tensor(a) for a in args[0]], [_np2Tensor(a) for a in args[1]]

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark = False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'

        self._set_filesystem(args.data_dir)
        self._get_imgs_path(args)
        self._set_dataset_length()

    def __getitem__(self, idx):
        lr, nei_lr, filename = self._load_file(idx)
        lr_tensor, nei_lr_tensor = np2Tensor2(lr, nei_lr,  rgb_range=self.args.rgb_range)
        return lr_tensor, nei_lr_tensor, filename

    def __len__(self):
        return self.dataset_length

    def _get_imgs_path(self, args):
        list_lr = self._scan()
        self.images_lr = list_lr

    def _set_dataset_length(self):
        self.dataset_length = len(self.images_lr[0]) #10

    def _scan(self):
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*' + self.ext[0]))
        )
        return [names_lr]

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, self.args.data_val_dir)

    def _get_index(self, idx):
        return (idx) #* 10 + 9 #* 10 + 9

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_lr = self.images_lr[0][idx]
        nei_lr = []
        lr = [imageio.imread(f_lr)]
        filename, _ = os.path.splitext(os.path.basename(f_lr))

        # nei_lr
        if self.args.frame == 5:
            if idx <= 1:
                nei_lr.append(imageio.imread(self.images_lr[0][0]))
                nei_lr.append(imageio.imread(self.images_lr[0][0]))
            else:
                nei_lr.append(imageio.imread(self.images_lr[0][idx-2]))
                nei_lr.append(imageio.imread(self.images_lr[0][idx-1]))
            if idx >= len(self.images_lr[0]) - 2:
                nei_lr.append(imageio.imread(self.images_lr[0][len(self.images_lr[0])-1]))
                nei_lr.append(imageio.imread(self.images_lr[0][len(self.images_lr[0])-1]))
            else:
                nei_lr.append(imageio.imread(self.images_lr[0][idx+1]))
                nei_lr.append(imageio.imread(self.images_lr[0][idx+2]))
        return lr, nei_lr, filename


