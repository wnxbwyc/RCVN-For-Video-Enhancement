import argparse
parser = argparse.ArgumentParser(description='video enhancement')

parser.add_argument('--n_threads', type=int, default=8,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--data_dir', type=str, default='data_path',
                    help='data_path')
parser.add_argument('--data_val_dir', type=str, default='val_video_png',
                    help='val_video_png')
# parser.add_argument('--data_val_compress', type=str, default='validation_fixed-QP_png',
#                    help='data_train_compress')
parser.add_argument('--data_val_compress', type=str, default='test_fixed-QP_png',
                   help='data_train_compress')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--model', help='model name', required=True)
parser.add_argument('--video_numbers', type=int, required=True)
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--video_index', default = 0,
                    help='video_index')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--n_resgroups', type=int, default=5,
                    help='number of residual groups')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--negval', type=float, default=0.2,
                    help='Negative value parameter for Leaky ReLU')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--frame', type=int, default=5,
                    help='number of frame to use')
args = parser.parse_args()
args.model = args.model.upper()
