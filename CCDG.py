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

from models.Conv1dBlock import Conv1dBlock
from models.Networks import Network_bearing, Network_fan
from datasets.load_bearing_data import ReadCWRU, ReadDZLRSB, ReadJNU, ReadPU, ReadMFPT, ReadUOTTAWA
from datasets.load_fan_data import ReadMIMII

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
from utils.GradientReserve import grad_reverse

# run code
# srun -w node3 --gres=gpu:1  /home/lsjia4/anaconda3/envs/pytorch/bin/python /home/lsjia4/MyFolder/fault_diagnosis/DGFDBenchmark/CCDG.py


with open(os.path.join(sys.path[0], 'config_files/CCDG_config.yaml')) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    print(configs)
    configs = DictObj(configs)

    if configs.use_cuda and torch.cuda.is_available():
        # set(configs,'device','cuda')
        configs.device='cuda'


class DomainContrastiveLoss(nn.Module):
    '''
    Copied from
    https://github.com/mohamedr002/CCDG
    '''
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device

    def forward(self, domains_features, domains_labels, temperature=0.7):
            # masking for the corresponding class labels.
        anchor_feature = domains_features
        anchor_feature = F.normalize(anchor_feature, dim=1)
        labels = domains_labels
        labels= labels.contiguous().view(-1, 1)
        # Generate masking for positive and negative pairs.
        mask = torch.eq(labels, labels.T).float().to(self.device)
        # Applying contrastive via product among the features
        # Measure the similarity between all samples in the batch
        # reiterate fact from Linear Algebra if u and v two vectors are normalised implies cosine similarity
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, anchor_feature.T), temperature)

        # for numerical stability
        # substract max value from the output
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # create inverted identity matrix with same shape as mask.
        # only diagnoal is zeros and all others are ones
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(anchor_feature.shape[0]).view(-1, 1).to(self.device), 0)

        # 上边这个完全可以使用： logits_mask = 1-torch.eye(mask.shape[0])

        # mask-out self-contrast cases
        # the diagnoal represent same samples similarity i=j>> we need to remove this
        # remove the true labels mask
        # all ones in mask would be only samples with same labels
        mask = mask * logits_mask

        # compute log_prob and remove the diagnal
        # remove same features from the normalized contrastive matrix
        # The denominoator of the equation samples
        exp_logits = torch.exp(logits) * logits_mask

        # substract the whole multiplications from the negative pair and positive pairs
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)
        zeros_idx = torch.where(mask_sum == 0)[0]
        mask_sum[zeros_idx] = 1

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = (- 1 * mean_log_prob_pos)
        loss = loss.mean()

        return loss



class CCDG(nn.Module):
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

        self.domain_contrastive_loss = DomainContrastiveLoss(configs).to(self.device)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

        if self.dataset_type=='bearing':
            self.model = Network_bearing(configs).to(self.device)
        elif self.dataset_type=='fan':
            self.model = Network_fan(configs).to(self.device)
        else:
            raise ValueError('The dataset_type should be bearing or fan!')
        self.optimizer = torch.optim.SGD(list(self.model.parameters()), lr=self.lr)
        self.weight_step = None

    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches]) # the length of the inner list is the number of the source domains (one machine is corresponding to a domain)
        labels = torch.cat([y for x, y in minibatches])
        x      = x.to(self.device)
        labels = labels.to(self.device)

        fv, logits = self.model(x)

        if  self.weight_step is None:
            self.weight_step = torch.ones(x.shape[0]).to(self.device)

        # ce_loss = F.cross_entropy(logits, labels)
        ce_loss = torch.mean(self.cross_entropy_loss(logits, labels)*self.weight_step)

        assert not torch.any(torch.isnan(fv)) #这个loss是施加在logits上的
        dc_loss = self.domain_contrastive_loss(fv, labels)

        loss = ce_loss + dc_loss
        self.optimizer.zero_grad()
        loss.backward()
        # with torch.autograd.detect_anomaly():
        #     loss.backward()
        self.optimizer.step()

        losses = {}
        losses['loss_total'] = loss.detach().cpu().data.numpy()
        losses['loss_ce'] = ce_loss.detach().cpu().data.numpy()
        losses['loss_id'] = dc_loss.detach().cpu().data.numpy()

        return losses

    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)

        loss_acc_result = {'loss_total': [], 'loss_ce':[], 'loss_id':[], 'acces':[]}

        for step in range(1, self.steps+1):
            self.train()
            self.current_step = step
            self.logger.info("================Step {}================".format(step))
            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)

            loss_acc_result['loss_total'].append(losses['loss_total'])
            loss_acc_result['loss_ce'].append(losses['loss_ce'])
            loss_acc_result['loss_id'].append(losses['loss_id'])
            # print(losses['loss_total'])
            # print(losses['loss_ce1'])
            # print(losses['loss_ce2'])

            self.logger.info('loss_total_train: \t {loss_total: .4f} \t loss_ce_train: \t {loss_ce: .4f} \t loss_id_train: \t {loss_id: .4f}'.format(loss_total=losses['loss_total'], loss_ce=losses['loss_ce'], loss_id=losses['loss_id']))

            #显示train_accuracy和test_accuracy
            if step % self.checkpoint_freq == 0 or step==self.steps:
                acc_results = self.test_model(test_loaders)
                loss_acc_result['acces'].append(acc_results)
                #################
                if self.use_domain_weight:
                    weight_step = torch.from_numpy(1-np.array(acc_results[1:])).type(torch.float32)
                    self.weight_step = weight_step.repeat((self.batch_size,1)).T.flatten(0).to(self.device)
                else:
                    self.weight_step = None

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

                label_pred = self.predict(x)
                y_pred_lst.extend(label_pred.detach().cpu().data.numpy())
                y_true_lst.extend(label_fault.cpu().numpy())

            acc_i, _, _, _ = cal_index(y_true_lst, y_pred_lst) #accracy of the i-th loader
            acc_results.append(acc_i)
        self.train()

        return acc_results

    def predict(self, x):
        with torch.no_grad():
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

    full_path_log = os.path.join('Output//CCDG//log_files', datasets_list[idx])
    if not os.path.exists(full_path_log):
        os.makedirs(full_path_log)

    full_path_rep = os.path.join('Output//CCDG//TuneReport', datasets_list[idx])
    if not os.path.exists(full_path_rep):
        os.makedirs(full_path_rep)


    currtime = str(time.time())[:10]
    logger = create_logger(full_path_log +'//log_file'+currtime)
    for i in range(1):
        model = CCDG(configs)
        for k, v in sorted(vars(configs).items()):
            logger.info('\t{}: {}'.format(k, v))

        loss_acc_result = model.train_model(train_minibatches_iterator, test_loaders_tgt+test_loaders_src, logger)



        loss_acc_result['loss_total'] = np.array(loss_acc_result['loss_total'])
        loss_acc_result['loss_ce'] = np.array(loss_acc_result['loss_ce'])
        loss_acc_result['loss_id'] = np.array(loss_acc_result['loss_id'])
        loss_acc_result['acces']   = np.array(loss_acc_result['acces'])


        # # save the loss curve and acc curve

        sio.savemat(full_path_log+'//loss_acc_result'+currtime+'.mat',loss_acc_result)
        gen_report = GenReport(full_path_rep)
        gen_report.write_file(configs=configs, test_item=None, loss_acc_result=loss_acc_result)
        gen_report.save_file(currtime)

        currtime = str(time.time())[:10]



if __name__ == '__main__':
    main(0, configs)
    # for i in range(5):
    #     main(i, configs)



# mfs_data = read_mfs.read_data_file()
# mfs_dataset_train = SimpleDataset(mfs_data['train'])
# batch_data = mfs_dataset_train[:5]
# x_out = the_model.forward(*batch_data)




# %%
