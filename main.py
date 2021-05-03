from models import make_model
from option import args
import data
from utils import test

models = make_model(args)
loader = data.Data(args)
loader_test = loader.loader_test
test(models, loader_test, args)


# python main.py --model rcvn_c --video_numbers 10 --save_results --frame 5