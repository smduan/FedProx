from ast import arguments
from utils_new import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--global_epoch',type=int,default=5)
parser.add_argument('--beta',default=0.05)
args = parser.parse_args()



if __name__ == "__main__":
    score_base = []
    for i in range(args.global_epoch):
        score_base.append(base_train(conf, "clinical", args.beta, 5, "./data/clinical/tmp/", "label",0))
    print(score_base)