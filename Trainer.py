import os
from math import ceil
import json

from torchtext.data import BucketIterator
from collections import defaultdict
from EpochRunner import EpochRunner
import torch

class Trainer:
    def __init__(self, model, args, word_i2s, optimizer_constructor, bert_optimizer_constructor, tester, EERuner):
        self.epochRunner = EpochRunner(word_i2s, EERuner, optimizer_constructor, bert_optimizer_constructor)
        self.model = model
        self.args = args
        self.tester = tester

    def eval(self, test_set):
        print("========test only=======")
        test_iter = BucketIterator(test_set, batch_size=self.args.batch, train=False, shuffle=False, device=self.model.device,
                                   sort_key=lambda x: len(x.TOKENS))
        with torch.no_grad():
            self.run_a_epoch("final test", test_iter, need_backward=False, epoch_num=0,
                             save_output=os.path.join(self.args.output_path, "check"),
                             max_step=ceil(len(test_set) / self.args.batch))
        print(self.model.argumentRoleClsLayer.arg_mask_init.tolist())
        print(self.model.argumentRoleClsLayer.arg_mask.tolist())

    def train(self, train_set, dev_set, test_set, epochs, other_testsets):

        # build batch on cpu
        train_iter = BucketIterator(train_set, batch_size=self.args.batch, train=True, shuffle=True, device=self.model.device,
                                    sort_key=lambda x: len(x.TOKENS))
        dev_iter = BucketIterator(dev_set, batch_size=self.args.batch, train=False, shuffle=False, device=self.model.device,
                                  sort_key=lambda x: len(x.TOKENS))
        test_iter = BucketIterator(test_set, batch_size=self.args.batch, train=False, shuffle=False, device=self.model.device,
                                   sort_key=lambda x: len(x.TOKENS))

        scores = -1.
        last_saving_at = -1
        now_bad = 0
        restart_used = 0
        print("\nStarting training...\n")
        prev_dev_score = 0

        best_test_score = -1
        test_score_dict = {
            "ed-det": defaultdict(list),
            "ed-cls": defaultdict(list),
            "emd-det": defaultdict(list),
            "emd-cls": defaultdict(list),
            "arg-cls": defaultdict(list)
                            }
        test_score_queue = {
            "ed-det": [], "ed-cls": [], "emd-det": [], "arg-cls": [], "emd-cls": []
        }
        for i in range(epochs):
            # Training Phrase
            print("Epoch {}, last saving at {}".format(i, last_saving_at))
            self.run_a_epoch("train", train_iter, need_backward=True, epoch_num=i, max_step=ceil(len(train_set) / self.args.batch))

            # Validation Phrase
            with torch.no_grad():
                dev_ed_f1, dev_edc_f1, dev_emd_f1, dev_emdc_f1, dev_arg_f1 = self.run_a_epoch("dev", dev_iter, need_backward=False, epoch_num=i, max_step=ceil(len(dev_set) / self.args.batch))
                print("=====================epoch end===================\n")
                # Early Stop
                dev_score = dev_ed_f1[0] + dev_edc_f1[0] + 0.5 * (dev_emd_f1[0] + dev_emdc_f1[0]) + 2 * dev_arg_f1[0]
                dev_score = (dev_score + prev_dev_score) / 2.0  # middle of current score and previous middle
                prev_dev_score = dev_score  # a new middle

                save_epoch_output = None
                if scores < dev_score:
                    scores = dev_score
                    # Move model parameters to CPU
                    self.model.save_model(os.path.join(self.args.output_path, "model.pt"))
                    last_saving_at = i
                    save_epoch_output = os.path.join(self.args.output_path, self.args.final_test_file)
                    print(">>>>>>>>>>>Save model at Epoch", i)
                    now_bad = 0
                else:
                    now_bad += 1
                    if now_bad >= self.args.earlystop:
                        pass
                        """
                        restart_used += 1
                        self.epochRunner.lr *= 0.7
                        print(">>>>>>>>>>>lr decays and best model is reloaded")
                        self.model.load_model(os.path.join(self.args.output_path, "model.pt"))
                        print("Restart in Epoch %d" % (i + 1))
                        now_bad = 0
                        """

                # Testing Phrase

                test_ed_f1, test_edc_f1, test_emd_f1, test_emdc_f1, test_arg_f1 = self.run_a_epoch("test", test_iter, need_backward=False, epoch_num=i, max_step=ceil(len(test_set) / self.args.batch),
                             save_output=save_epoch_output)
                test_score = test_ed_f1[0] + test_edc_f1[0] + 0.5 * (test_emd_f1[0] + test_emdc_f1[0]) + 2 * test_arg_f1[0]
                test_score_dict["ed-det"][str(int(test_ed_f1[0]))].append(1)
                test_score_queue["ed-det"].append(str(int(test_ed_f1[0])))
                test_score_dict["ed-cls"][str(int(test_edc_f1[0]))].append(1)
                test_score_queue["ed-cls"].append(str(int(test_edc_f1[0])))
                test_score_dict["emd-det"][str(int(test_emd_f1[0]))].append(1)
                test_score_queue["emd-det"].append(str(int(test_emd_f1[0])))
                test_score_dict["emd-cls"][str(int(test_emdc_f1[0]))].append(1)
                test_score_queue["emd-cls"].append(str(int(test_emdc_f1[0])))
                test_score_dict["arg-cls"][str(int(test_arg_f1[0]))].append(1)
                test_score_queue["arg-cls"].append(str(int(test_arg_f1[0])))
                if best_test_score < test_score:
                    print("============Best Model can be saved at Epoch ", i)
                    best_test_score = test_score
                    self.model.save_model(os.path.join(self.args.output_path, "best.pt"))
                if i % 10 == 0:
                    self.percentage_of_test_score(epoch_num=i, test_score_dict=test_score_dict, test_score_queue=test_score_queue)

        print("Train Finished! output the test_res")
        self.percentage_of_test_score(epoch_num=epochs, test_score_dict=test_score_dict, test_score_queue=test_score_queue)
        # Testing Phrase: load the best model if exist
        if os.path.exists(os.path.join(self.args.output_path, "model.pt")):
            print("loaded exist best model successfully")
            self.model.load_model(os.path.join(self.args.output_path, "model.pt"))
        self.run_a_epoch("final test", test_iter, need_backward=False, epoch_num=0,
                         save_output=os.path.join(self.args.output_path,self.args.final_test_file),
                         max_step=ceil(len(test_set) / self.args.batch))

        for name, additional_test_set in other_testsets.items():
            additional_test_iter = BucketIterator(additional_test_set, batch_size=self.args.batch, train=False, shuffle=True,
                                                  device=-1,
                                                  sort_key=lambda x: len(x.POSTAGS))

            self.run_a_epoch(name, additional_test_iter, need_backward=False, epoch_num=-11, max_step=ceil(len(additional_test_set) / self.args.batch))

        print("Training Done!")

    def run_a_epoch(self, step, data_iter, need_backward, epoch_num, max_step,save_output=None):
        ed_loss, edc_loss, emd_loss, emdc_loss, arg_loss,\
            edp, edr, edf, edcp, edcr, edcf, emdp, emdr, emdf, emdcp, emdcr, emdcf, \
            argp, argr, argf = self.epochRunner.run_one_epoch(data_iter=data_iter,
                                                              model=self.model,
                                                              need_backward=need_backward,
                                                              epoch_num=epoch_num,
                                                              MAX_STEP=max_step,
                                                              tester=self.tester,
                                                              hyps=self.model.hyperparams,
                                                              device=self.model.device,
                                                              save_output=save_output,
                                                              maxnorm=self.args.hps["maxnorm"]
                                                              )
        total_loss = edc_loss + ed_loss + emdc_loss + emd_loss
        print("Epoch {}'s {} total_loss: {}".format(epoch_num, step, total_loss))
        self.phase_record(step, "detect", "ed", ed_loss, edp, edr, edf, epoch_num, save_output)
        self.phase_record(step, "cls", "ed", edc_loss, edcp, edcr, edcf, epoch_num, save_output)
        #self.args.writer["train"].add_scalar('{}/ed/f1'.format(step), (edcf + edf) //2, epoch_num) if not save_output else None
        self.args.writer["train"].add_scalar('{}/ed/loss'.format(step), (ed_loss + edc_loss), epoch_num) if not save_output else None

        self.phase_record(step, "detect", "emd", emd_loss, emdp, emdr, emdf, epoch_num, save_output)
        self.phase_record(step, "cls", "emd", emdc_loss, emdcp, emdcr, emdcf, epoch_num, save_output)
        #self.args.writer["train"].add_scalar('{}/emd/f1'.format(step), (emdcf + emdf) //2, epoch_num) if not save_output else None
        self.args.writer["train"].add_scalar('{}/emd/loss'.format(step), (emd_loss + emdc_loss), epoch_num) if not save_output else None

        self.phase_record(step, "cls", "arg", arg_loss, argp, argr, argf, epoch_num, save_output)
        print()
        return edf, edcf, emdf, emdcf, argf

    def phase_record(self, step, phase, type, loss, p, r, f, epoch_num, save_out=None):
        print("Epoch {}'s {} Phase:{}-{} p: {} r: {} f1: {}, loss={}".format(epoch_num, step, type, phase, p[0], r[0], f[0], loss)) if r[0] > 0. else None
        #print("Epoch {}'s {} Phase:{}-{} p: {} r: {} f1: {} ---Sklearn metrics".format(epoch_num, step, type, phase, p[1], r[1], f[1])) if r[0] > 0. else None
        if not save_out:
            self.args.writer[phase].add_scalar('{}/{}/loss'.format(step, type), loss, epoch_num)
            self.args.writer[phase].add_scalar('{}/{}/p'.format(step, type), p[0], epoch_num)
            self.args.writer[phase].add_scalar('{}/{}/r'.format(step, type), r[0], epoch_num)
            self.args.writer[phase].add_scalar('{}/{}/f1'.format(step, type), f[0], epoch_num)

    def percentage_of_test_score(self, epoch_num, test_score_dict, test_score_queue):
        all_percentage = {type_: {key:
                                      round(100. * sum(value)/(epoch_num + 1), 2) for key, value in dict_.items()}
                          for type_, dict_ in test_score_dict.items()}

        last_100e_score_dict = {
            "ed-det": defaultdict(int),
            "ed-cls": defaultdict(int),
            "emd-det": defaultdict(int),
            "emd-cls": defaultdict(int),
            "arg-cls": defaultdict(int)
        }
        for key in all_percentage:
            for score in all_percentage[key]:
                cnt = test_score_queue[key][-100:].count(score)
                if not cnt: continue
                last_100e_score_dict[key][score] = cnt

        last_100e_percentage = {type_: {key:
                                      round(100. * value/100, 0) for key, value in dict_.items()}
                          for type_, dict_ in last_100e_score_dict.items()}
        print("Epoch 0 to {}'s test statistic report is {}".format(epoch_num, json.dumps(all_percentage, indent=2)))
        print("Last 100  Epoch's test statistic report is :"
              "\ned-det : {}".format(sorted(last_100e_percentage["ed-det"].items())),
              "\ned-cls : {}".format(sorted(last_100e_percentage["ed-cls"].items())),
              "\nemd-det : {}".format(sorted(last_100e_percentage["emd-det"].items())),
              "\nemd-cls : {}".format(sorted(last_100e_percentage["emd-cls"].items())),
              "\narg-cls : {}".format(sorted(last_100e_percentage["arg-cls"].items())))