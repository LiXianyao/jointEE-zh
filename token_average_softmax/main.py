#-*-encoding:utf-8-*-#

import sys
sys.path.append("..")
from util.Configure import *

from transformers import AutoConfig, BertTokenizer
from network.models.Joint3EE import Joint3EEModel
from EERunner import EERunner
import os, sys

class Runner(EERunner):
    def __init__(self, conf, device=None):
        EERunner.__init__(self, conf, device)
        self.bertTokenizer = BertTokenizer.from_pretrained(self.arg.bert_model,
                                                       cache_dir=os.path.join(self.arg.home_path, "pretrain-bert", self.arg.bert_model))
        self.bertConfig = AutoConfig
        self.modelClass = Joint3EEModel
        dirS = os.path.join(self.arg.home_path, "baselineStdout", "zh_token_average_softmax")
        if not os.path.exists(dirS):
            os.makedirs(dirS)
        self.stdoutFile = open(os.path.join(dirS, self.arg.final_test_file), "w", encoding="utf-8")
        sys.stdout = self.stdoutFile
        sys.stderr = self.stdoutFile

    def __del__(self):
        self.stdoutFile.close()

if __name__ == "__main__":
    confs = Configure()
    runner = Runner(confs)
    runner.run()
    #runner.run(Train=False, Test=True)
