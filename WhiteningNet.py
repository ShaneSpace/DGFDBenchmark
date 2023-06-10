import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
# from torch.autograd import Function
# from torchvision import models
from torchsummary import summary


import numpy as np
import yaml
import itertools
import copy
import random
import os
import time
import sys
from builtins import object
from logging import exception
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, freqz, hilbert
import pickle
import matplotlib.pyplot as plt
import math


from models.Conv1dBlock import Conv1dBlock
from models.Networks import Network_bearing, Network_fan
from datasets.load_bearing_data import ReadCWRU, ReadDZLRSB, ReadJNU, ReadPU, ReadMFPT, ReadUOTTAWA
from datasets.load_fan_data import ReadMIMII

# self-made utils
from utils.DictObj import DictObj
from utils.AverageMeter import AverageMeter
from utils.CalIndex import cal_index
# from utils.SetSeed import set_random_seed
from utils.CreateLogger import create_logger
from utils.SimpleLayerNorm import LayerNorm
from utils.TuneReport import GenReport
from utils.DatasetClass import InfiniteDataLoader, SimpleDataset
# from utils.SignalTransforms import AddGaussianNoise, RandomScale, MakeNoise, Translation
# from utils.LMMD import LMMDLoss
# from utils.GradientReserve import grad_reverse

# run code
# srun -w node3 --gres=gpu:1  /home/lsjia4/anaconda3/envs/pytorch/bin/python /home/lsjia4/MyFolder/fault_diagnosis/DGFDBenchmark/WhiteningNet.py





with open(os.path.join(sys.path[0], 'config_files/whiteningnet_config.yaml')) as f:
    '''
    link: https://zetcode.com/python/yaml/#:~:text=YAML%20natively%20supports%20three%20basic,for%20YAML%3A%20PyYAML%20and%20ruamel.
    '''
    configs = yaml.load(f, Loader=yaml.FullLoader)
    print(configs) #yaml库读进来的是字典dict格式，需要用DictObj将dict转换成object
    configs = DictObj(configs)

    if configs.use_cuda and torch.cuda.is_available():
        # set(configs,'device','cuda')
        configs.device='cuda'

#%% model


class CausalLoss(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.num_classes = configs.num_classes

    def forward(self, flatten_features, data_label):
        list_category = [[] for i in range(self.num_classes)]
        # list_label = [[] for i in range(self.num_classes)] # for debug
        for i, fv in zip(data_label, flatten_features):
            fv = torch.reshape(fv, (1, fv.size(0)))
            list_category[i].append(fv)
            # list_label[i].append(i) # for debug
        # self.list_label = list_label# for debug

        total_causal_loss = 0
        for i in range(self.num_classes):
            fm_i = torch.cat(tuple(list_category[i]), dim=0) # convert the feature vector listinto a single tensor(matrix)
            # print(fm_i.shape)
            fm_i_mean = torch.mean(fm_i, dim=0, keepdim=True)
            causal_i = torch.sum(torch.mean((fm_i - fm_i_mean).pow(2), dim=0))
            total_causal_loss = total_causal_loss + causal_i

        return total_causal_loss

class WhitenNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.dataset_type = configs.dataset_type

        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        self.use_domain_weight = configs.use_domain_weight
        if self.dataset_type == 'bearing':
            self.model = Network_bearing(configs).to(self.device)
        elif self.dataset_type == 'fan':
            self.model = Network_fan(configs).to(self.device)

        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.lr)

        self.cl_loss = nn.CrossEntropyLoss(reduction='none')
        self.causal_loss = CausalLoss(configs)

        self.lamda_causal = configs.lamda_causal
        self.num_domains = len(configs.datasets_src)
        self.weight_step = None
        self.use_learning_rate_sheduler = configs.use_learning_rate_sheduler



    def adjust_learning_rate(self, step):
        """
        Decay the learning rate based on schedule
        https://github.com/facebookresearch/moco/blob/main/main_moco.py
        """
        lr = self.lr
        if self.configs.cos:  # cosine lr schedule
            lr *= 0.5 * (1.0 + math.cos(math.pi * step / self.steps))
        else:  # stepwise lr schedule
            for milestone in self.configs.schedule:
                lr *= 0.5 if step >= milestone else 1.0
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches]) # the length of the inner list is the number of the source domains (one machine is corresponding to a domain)
        # debug
        # x_shape = [x.shape[0] for x,y in minibatches]
        # print('The shape of X',x_shape)
        labels = torch.cat([y for x, y in minibatches])
        x      = x.to(self.device)
        labels = labels.to(self.device)

        feature_vectors, logits = self.model(x)

        if self.use_domain_weight:
            if  self.weight_step is None:
                self.weight_step = torch.ones(x.shape[0]).to(self.device)
            else:
                ce_values  = self.cl_loss(logits, labels)
                ce_values_2d = torch.reshape(ce_values, (self.num_domains, self.batch_size))
                ce_value_domain = torch.mean(ce_values_2d,dim=1)
                ce_value_sum = torch.sum(ce_value_domain)
                weight_step = 1 + ce_value_domain/ce_value_sum
                self.weight_step = weight_step.repeat((self.batch_size,1)).T.flatten(0).to(self.device)
        else:
            self.weight_step = torch.ones(x.shape[0]).to(self.device)

        cl_loss = torch.mean(self.cl_loss(logits, labels)*self.weight_step)
        causal_loss = self.causal_loss(feature_vectors, labels)
        total_loss = cl_loss + self.lamda_causal*causal_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        loss_cl = cl_loss.detach().cpu().data.numpy()
        loss_causal = causal_loss.detach().cpu().data.numpy()


        losses={}

        losses['cl'] = loss_cl
        losses['causal'] = loss_causal

        return losses

    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)

        # loss_acc_result = {'loss_cc': [], 'loss_ct':[], 'loss_cl':[], 'acces':[]}
        loss_acc_result = {'loss_causal': [], 'loss_cl':[], 'acces':[]}

        for step in range(1, self.steps+1):
            self.train()
            self.current_step = step
            self.logger.info("================Step {}================".format(step))
            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)
            # self.scheduler.step()
            if self.use_learning_rate_sheduler:
                self.adjust_learning_rate(self.current_step)

            # loss_acc_result['loss_cc'].append(losses['cc'])
            # loss_acc_result['loss_ct'].append(losses['ct'])
            loss_acc_result['loss_cl'].append(losses['cl'])
            loss_acc_result['loss_causal'].append(losses['causal'])

            # self.logger.info('loss_cl_train: \t {loss_cl: .4f} \t loss_cc_train: \t {loss_cc: .4f} \t loss_ct_train: \t {loss_ct: .4f} \t loss_cl_train: \t {loss_cl: .4f}'.format(loss_cl=losses['cl'], loss_cc=losses['cc'], loss_ct=losses['ct']))
            self.logger.info('loss_cl_train: \t {loss_cl: .4f} \t loss_proto_train: \t {loss_causal: .4f} '.format(loss_cl=losses['cl'], loss_causal=losses['causal']))


            #显示train_accuracy和test_accuracy
            if step % self.checkpoint_freq == 0 or step==self.steps:
                acc_results = self.test_model(test_loaders)
                loss_acc_result['acces'].append(acc_results)
                # 2023/03/19
                # this part can be used to calculate the weight of domains
                # if self.use_domain_weight:
                #     weight_step = torch.from_numpy(1-np.array(acc_results[1:])).type(torch.float32)
                #     self.weight_step = weight_step.repeat((self.batch_size,1)).T.flatten(0).to(self.device)
                # else:
                #     self.weight_step = None

                self.logger.info('tgt_train_acc: \t {src_train_acc: .4f}'.format(src_train_acc=acc_results[0]))
                for i in range(1,len(acc_results)):
                    self.logger.info('src_train_acc: \t {tgt_train_acc: .4f}'.format(tgt_train_acc=acc_results[i]))

        return loss_acc_result

    def test_model(self, loaders):
        self.eval()
        num_loaders = len(loaders)
        acc_results = []
        for i in range(num_loaders):
            the_loader = loaders[i]
            y_pred_lst = []
            y_true_lst = []
            for j, batched_data in enumerate(the_loader):
                x, label_fault = batched_data
                x = x.to(self.device)
                label_fault = label_fault.to(self.device)
                # # debug
                # print(j)
                # print(x.shape)

                label_pred = self.predict(x)
                y_pred_lst.extend(label_pred.detach().cpu().data.numpy())
                y_true_lst.extend(label_fault.cpu().numpy())

            acc_i, _, _, _ = cal_index(y_true_lst, y_pred_lst) #accracy of the i-th loader
            acc_results.append(acc_i)
        self.train()

        return acc_results

    def predict(self,x):
        # print(x.shape)
        _, logits = self.model(x)

        return torch.max(logits, dim=1)[1]


def main(idx, configs):
    if configs.dataset_type == 'bearing':
        # datasets_list = ['CWRU','JNU','UOTTAWA','MFPT','PU','DZLRSB']
        # datasets_list = ['CWRU','JNU','UOTTAWA','MFPT','DZLRSB']
        datasets_list = ['CWRU','UOTTAWA','MFPT','DZLRSB']
    elif configs.dataset_type=='fan':
        configs.num_classes = 2
        configs.batch_size  = 32
        configs.steps       = 100
        if configs.fan_section=='sec00':
            datasets_list =  ['W','X','Y','Z']
            section='00'
        elif configs.fan_section=='sec01':
            datasets_list=['A','B','C']
            section='01'
        else:
            datasets_list=['L1','L2','L3','L4']
            section = '02'
    else:
        raise ValueError('The dataset_type should be bearing or fan!')

    dataset_idx = list(range(len(datasets_list)))
    tgt_idx = [idx]
    src_idx = [i for i in dataset_idx if not tgt_idx.__contains__(i)]
    datasets_tgt = [datasets_list[i] for i in tgt_idx]
    datasets_src = [datasets_list[i]  for i in src_idx]
    configs.datasets_tgt = datasets_tgt
    configs.datasets_src = datasets_src


    if configs.dataset_type == 'bearing':
        datasets_object_src = [eval('Read'+i+'(configs)') for i in datasets_src]
    elif configs.dataset_type=='fan':
        datasets_object_src = [ReadMIMII(i, section=section, configs=configs) for i in datasets_src]
    train_test_loaders_src = [dataset.load_dataloaders() for dataset in datasets_object_src]
    train_loaders_src = [train for train,test in train_test_loaders_src]
    test_loaders_src  = [test for train,test in train_test_loaders_src]

    if configs.dataset_type == 'bearing':
        datasets_object_tgt = [eval('Read'+i+'(configs)') for i in datasets_tgt]
    elif configs.dataset_type=='fan':
        datasets_object_tgt = [ReadMIMII(i, section=section, configs=configs) for i in datasets_tgt]
    train_test_loaders_tgt = [dataset.load_dataloaders() for dataset in datasets_object_tgt]
    # train_loaders_tgt = [train for train,test in train_test_loaders_tgt]
    test_loaders_tgt  = [test  for train,test in train_test_loaders_tgt]

    train_minibatches_iterator = zip(*train_loaders_src)

    full_path_log = os.path.join('Output//WhiteningNet//log_files', datasets_list[idx])
    if not os.path.exists(full_path_log):
        os.makedirs(full_path_log)

    full_path_rep = os.path.join('Output//WhiteningNet//TuneReport', datasets_list[idx])
    if not os.path.exists(full_path_rep):
        os.makedirs(full_path_rep)




    currtime = str(time.time())[:10]
    logger = create_logger(full_path_log +'//log_file'+currtime)
    for i in range(1):
        model = WhitenNet(configs)

        for k, v in sorted(vars(configs).items()):
            logger.info('\t{}: {}'.format(k, v))

        loss_acc_result = model.train_model(train_minibatches_iterator, test_loaders_tgt+test_loaders_src, logger)


        loss_acc_result['loss_cl'] = np.array(loss_acc_result['loss_cl'])
        loss_acc_result['loss_causal'] = np.array(loss_acc_result['loss_causal'])
        loss_acc_result['acces']   = np.array(loss_acc_result['acces'])


        # # save the loss curve and acc curve
        # sio.savemat('WhitenNet_reproduced//log_files//loss_acc_result'+currtime+'.mat',loss_acc_result)
        # gen_report = GenReport('WhitenNet_reproduced//TuneReport//')
        sio.savemat(full_path_log+'//loss_acc_result'+currtime+'.mat',loss_acc_result)
        gen_report = GenReport(full_path_rep)
        gen_report.write_file(configs=configs, test_item=None, loss_acc_result=loss_acc_result)
        gen_report.save_file(currtime)

        # torch.save(model.to('cpu'),'IDEA_test2//saved_models//model'+currtime+'.pt')
        currtime = str(time.time())[:10]



if __name__ == '__main__':
    main(0, configs)
    # for i in range(5):
    #     main(i, configs)

