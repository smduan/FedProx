from ast import arguments
from func import *
import argparse
from conf import conf

parser = argparse.ArgumentParser()
parser.add_argument('--global_epoch',type=int,default=conf['global_epochs'])
parser.add_argument('--beta',default=conf['beta'])
args = parser.parse_args()



if __name__ == "__main__":
    score_base = []
    for i in range(args.global_epoch):
        score_base.append(base_train(conf, "clinical", args.beta, 5, "./data/clinical/tmp/", "label",0))
    print(score_base)