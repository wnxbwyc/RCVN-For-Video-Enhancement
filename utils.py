import time
import torch
import data
from tqdm import tqdm
from decimal import Decimal
import math
import os
import imageio

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def PIM(lr, neilr, model, stride = 62, re_c = 5):
    _, c, h, w = lr[0].size()
    lr = lr[0]
    neilr = neilr[0]
    sr = torch.zeros_like(lr)
    sr_counts = torch.zeros_like(lr)

    # modify higher is better
    modify = 1
    # h_num = min(math.ceil(h / 72 * modify), h // 72)
    # w_num = min(math.ceil(w / 72 * modify), w // 72)
    h_num = min(math.ceil( (h - 72) / stride * modify) + 1, (h - 72) // stride + 1)
    w_num = min(math.ceil( (w - 72) / stride * modify) + 1, (w - 72) // stride + 1)
    h_num2 = h - 72 - (h_num - 1) * stride
    w_num2 = w - 72 - (w_num - 1) * stride
 
    if((72 - stride) < (2 * re_c)):
        re_c = (72 - stride) // 2
    # top-left
    sr_new_list = []
    for row in range(h_num):
        for col in range(w_num):
            r1 = row * stride
            c1 = col * stride
            sr_new_list.append(lr[: ,: , r1: r1 + 72, c1: c1 + 72])
            if((row != 0)&(col != 0)):
                sr_counts[: , : , r1 + re_c: r1 + 72 - re_c , c1 + re_c : c1 + 72 - re_c] += 1
            else:
                sr_counts[: , : , r1: r1 + 72, c1: c1 + 72] += 1
    sr_new = torch.cat(sr_new_list, dim = 0)
    neilr2 = []
    for index in range(len(neilr)):
        neilr2_list = []
        for row in range(h_num):
            for col in range(w_num):
                r1 = row * stride
                c1 = col * stride
                # neilr2_list.append(find_min_mae(lr, neilr[index], r1, c1))
                neilr2_list.append(neilr[index][: ,: , r1: r1 + 72, c1: c1 + 72])
        neilr2_list = torch.cat(neilr2_list, dim = 0)
        neilr2.append(neilr2_list)
    sr_new = model(sr_new, neilr2)
    sr_new = torch.cat(torch.cat(sr_new.chunk(h_num, dim=0), dim=2).chunk(w_num, dim=0),dim=3)
    for row in range(h_num):
        for col in range(w_num):
            r1 = row * stride
            c1 = col * stride
            if((row != 0)&(col != 0)):
                sr[:, :, r1 + re_c: r1 + 72 - re_c, c1 + re_c: c1 + 72 - re_c] += sr_new[: ,: ,row * 72 + re_c: (row + 1) * 72 - re_c, col * 72 + re_c:(col + 1) * 72 - re_c]
            else:
                sr[:, :, r1: r1 + 72, c1: c1 + 72] += sr_new[: ,: ,row * 72: (row + 1) * 72 , col * 72:(col + 1) * 72]

    # top-right
    sr_new_list = []
    for row in range(h_num):
        for col in range(w_num):
            sr_new_list.append(lr[: ,: , row * stride: row * stride + 72, w_num2 + col * stride: w_num2 + col * stride + 72])
            r1 = row * stride
            c1 = col * stride
            if((row != 0)&(col != (w_num - 1))):
                sr_counts[: ,: , r1 + re_c: r1 + 72 - re_c, w_num2 + c1 + re_c: w_num2 + c1 + 72 - re_c] += 1
            else:
                sr_counts[: ,: , r1: r1 + 72, w_num2 + c1: w_num2 + c1 + 72] += 1
    sr_new = torch.cat(sr_new_list, dim = 0)
    neilr2 = []
    for index in range(len(neilr)):
        neilr2_list = []
        for row in range(h_num):
            for col in range(w_num):
                r1 = row * stride
                c1 = col * stride
                # neilr2_list.append(find_min_mae(lr, neilr[index], r1, w_num2 + c1))
                neilr2_list.append(neilr[index][: ,: , r1: r1 + 72, w_num2 + c1: w_num2 + c1 + 72])
        neilr2_list = torch.cat(neilr2_list, dim = 0)
        neilr2.append(neilr2_list)
    sr_new = model(sr_new, neilr2)
    sr_new = torch.cat(torch.cat(sr_new.chunk(h_num, dim=0), dim=2).chunk(w_num, dim=0),dim=3)
    for row in range(h_num):
        for col in range(w_num):
            r1 = row * stride
            c1 = col * stride
            if((row != 0)&(col != (w_num - 1))):
                sr[: ,: , r1 + re_c: r1 + 72 - re_c, w_num2 + c1 + re_c: w_num2 + c1 + 72 - re_c] += sr_new[: ,: ,row * 72 + re_c: (row + 1) * 72 -re_c, col * 72 + re_c:(col + 1) * 72 - re_c]
            else:
                sr[: ,: , r1: r1 + 72, w_num2 + c1: w_num2 + c1 + 72] += sr_new[: ,: ,row * 72: (row + 1) * 72 , col * 72:(col + 1) * 72]

    # bottom-left
    sr_new_list = []
    for row in range(h_num):
        for col in range(w_num):
            r1 = row * stride
            c1 = col * stride
            sr_new_list.append(lr[: ,: , h_num2 + r1: h_num2 + r1 + 72, c1: c1 + 72])
            if((row != (h_num - 1))&(col != 0)):
                sr_counts[: ,: , h_num2 + r1 + re_c: h_num2 + r1 + 72 - re_c, c1 + re_c: c1 + 72 - re_c] += 1
            else:
                sr_counts[: ,: , h_num2 + r1: h_num2 + r1 + 72, c1: c1 + 72] += 1
    sr_new = torch.cat(sr_new_list, dim = 0)
    neilr2 = []
    for index in range(len(neilr)):
        neilr2_list = []
        for row in range(h_num):
            for col in range(w_num):
                r1 = row * stride
                c1 = col * stride
                # neilr2_list.append(find_min_mae(lr, neilr[index], h_num2 + r1, c1))
                neilr2_list.append(neilr[index][: ,: , h_num2 + r1: h_num2 + r1 + 72, c1: c1 + 72])
        neilr2_list = torch.cat(neilr2_list, dim = 0)
        neilr2.append(neilr2_list)
    sr_new = model(sr_new, neilr2)
    sr_new = torch.cat(torch.cat(sr_new.chunk(h_num, dim=0), dim=2).chunk(w_num, dim=0),dim=3)
    for row in range(h_num):
        for col in range(w_num):
            r1 = row * stride
            c1 = col * stride
            if((row != (h_num - 1))&(col != 0)):
                sr[: ,: , h_num2 + r1 + re_c: h_num2 + r1 + 72 - re_c, c1 + re_c: c1 + 72 - re_c] += sr_new[: ,: ,row * 72 + re_c: (row + 1) * 72 - re_c , col * 72 + re_c:(col + 1) * 72 - re_c]
            else:
                sr[: ,: , h_num2 + r1: h_num2 + r1 + 72, c1: c1 + 72] += sr_new[: ,: ,row * 72: (row + 1) * 72 , col * 72:(col + 1) * 72]

    # bottom-right
    sr_new_list = []
    for row in range(h_num):
        for col in range(w_num):
            r1 = row * stride
            c1 = col * stride
            sr_new_list.append(lr[: ,: , h_num2 + r1: h_num2 + r1 + 72, w_num2 + c1: w_num2 + c1 + 72])
            if((row != (h_num - 1))&(col != (w_num - 1))):
                sr_counts[: ,: , h_num2 + r1 + re_c: h_num2 + r1 + 72 - re_c, w_num2 + c1 + re_c: w_num2 + c1 + 72 - re_c] += 1
            else:
                sr_counts[: ,: , h_num2 + r1: h_num2 + r1 + 72, w_num2 + c1: w_num2 + c1 + 72] += 1
    sr_new = torch.cat(sr_new_list, dim = 0)
    neilr2 = []
    for index in range(len(neilr)):
        neilr2_list = []
        for row in range(h_num):
            for col in range(w_num):
                r1 = row * stride
                c1 = col * stride
                # neilr2_list.append(find_min_mae(lr, neilr[index], h_num2 + r1, w_num2 + c1))
                neilr2_list.append(neilr[index][: ,: , h_num2 + r1: h_num2 + r1 + 72, w_num2 + c1: w_num2 + c1 + 72])
        neilr2_list = torch.cat(neilr2_list, dim = 0)
        neilr2.append(neilr2_list)
    sr_new = model(sr_new, neilr2)
    sr_new = torch.cat(torch.cat(sr_new.chunk(h_num, dim=0), dim=2).chunk(w_num, dim=0),dim=3)
    for row in range(h_num):
        for col in range(w_num):
            r1 = row * stride
            c1 = col * stride
            if((row != (h_num - 1))&(col != (w_num - 1))):
                sr[: ,: , h_num2 + r1 + re_c: h_num2 + r1 + 72 - re_c, w_num2 + c1 + re_c: w_num2 + c1 + 72 - re_c] += sr_new[: ,: ,row * 72 + re_c: (row + 1) * 72 - re_c, col * 72 + re_c:(col + 1) * 72 - re_c]
            else:
                sr[: ,: , h_num2 + r1: h_num2 + r1 + 72, w_num2 + c1: w_num2 + c1 + 72] += sr_new[: ,: ,row * 72: (row + 1) * 72 , col * 72:(col + 1) * 72]

    sr = sr / sr_counts
    sr = quantize(sr, 255)
    return sr

def prepare2(*data):
    from option import args
    device = torch.device('cpu' if args.cpu else 'cuda')
    if len(data) == 4:
        return [a.to(device) for a in data[0]], data[1].to(device),[a.to(device) for a in data[2]],[a.to(device) for a in data[3]]
    elif len(data) == 3:
        return [a.to(device) for a in data[0]], data[1].to(device), [a.to(device) for a in data[2]]
    elif len(data) > 1:
        return [a.to(device) for a in data[0]], data[-1].to(device)
    return [a.to(device) for a in data[0]],

def save_results(model_name, filename, sr, video_index):
    apath = '{}/test_results/{}'.format(model_name, str(video_index + 1).zfill(3))
    if not os.path.exists(apath):
        os.makedirs(apath)
    filename = os.path.join(apath, filename[0])
    if(len(sr.shape)==4):
        sr = sr[0]
    normalized = sr.data.mul(255 / 255)
    ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
    imageio.imwrite('{}.png'.format(filename), ndarr)

def test(models, loader_test, opt):
    models.eval()
    image_num = 0
    timer_test = timer()
    timer_data, timer_model = timer(), timer()
    
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    runtime = 0

    with torch.no_grad():
        timer_data.tic()
        for video_index in range(opt.video_numbers):
            opt.video_index = video_index
            loader_test2 = data.Data(opt).loader_test
            tqdm_test = tqdm(loader_test2, ncols=80)
            for _, (lr, neilr, filename) in enumerate(tqdm_test):
                if os.path.exists(os.path.join('{}/test_results/{}'.format(opt.model.upper(), str(video_index + 1).zfill(3)), filename[0] + '.png')):
                    continue
                lr,  = prepare2(lr)
                neilr = prepare2(neilr)
                # torch.cuda.synchronize()
                timer_data.hold()
                timer_model.tic()
                # start.record()
                sr = PIM(lr, neilr, models)
                # end.record()
                # torch.cuda.synchronize()
                # runtime += start.elapsed_time(end)
                timer_model.hold()

                save_results(opt.model.upper(), filename, sr, video_index)
                timer_data.tic()
                image_num += len(loader_test)
        print('data_test: {}'.format(opt.data_val_dir))
        print('Total time: {:.3f}s'.format(timer_test.toc()))
        # print('running_time : {:.6f}s'.format(runtime / 1000.0 / 1.0 / image_num))
        print('data time: {:.3f}s , model time: {:.3f}s'.format(timer_data.release(), timer_model.release()))


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0