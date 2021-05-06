# RCVN-For-Video-Enhancement

# NTIRE 2021 video/multi-frame challenges

[Official page](https://github.com/RenYang-home/NTIRE21_VEnh)

## Quality enhancement of heavily compressed videos: Track 1 Fixed QP, Fidelity

### The test results of Track 1:

|rank | PSNR | rank | PSNR |
|---- | ---- |---- | ---- |
| 1  |32.52| 7  |31.75|
| 2  |32.49| 8  |31.65|
| 3  |32.04| 9  |31.62|
| 4  |31.90|**Ours** |31.59|
| 5  |31.86| 11  |31.37|
| 6  |31.78| 12  |31.13|

---

## We implemented our methods with 
* python 3.8.8
* pytorch 1.8.0
* cuda 11.1 
* cudnn 8.0.5.

In our machine, it cost about 13.72 seconds to generation a png.

---

### For test, you can follow the steps below.
#### 1. First set full compressed pngs in 

/data_path/val_video_png/test_fixed-QP_png/001/xxx.png

/data_path/val_video_png/test_fixed-QP_png/002/xxx.png

...

/data_path/val_video_png/test_fixed-QP_png/010/xxx.png

We have set 10 pngs in the 001 folder for an example.

#### 2. Run the following code to generation enhanced pngs:

```bash
python main.py --model rcvn_c --video_numbers 10 --save_results --frame 5
```

#### 3. The full enhanced pngs(or 10 enhanced example pngs) results will be generated in 

/RCVN_C/test_results/001/xxx.png

...

...
