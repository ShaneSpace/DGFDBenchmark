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
from models.Networks import FeatureExtractor_iedg_bearing, FaultClassifier_iedg_bearing, Discriminator_iedg_bearing, FeatureExtractor_iedg_fan,FaultClassifier_iedg_fan, Discriminator_iedg_fan
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
from utils.GradientReserve import grad_reverse

# run code
# srun -w node3 --gres=gpu:1  /home/lsjia4/anaconda3/envs/pytorch/bin/python /home/lsjia4/MyFolder/fault_diagnosis/DGFDBenchmark/IEDGNet.py


with open(os.path.join(sys.path[0], 'config_files/IEDGNet_config.yaml')) as f:
    '''
    link: https://zetcode.com/python/yaml/#:~:text=YAML%20natively%20supports%20three%20basic,for%20YAML%3A%20PyYAML%20and%20ruamel.
    '''
    configs = yaml.load(f, Loader=yaml.FullLoader)
    print(configs) #yaml库读进来的是字典dict格式，需要用DictObj将dict转换成object
    configs = DictObj(configs)

    if configs.use_cuda and torch.cuda.is_available():
        # set(configs,'device','cuda')
        configs.device='cuda'

    # 2023-01-26
    # Note: "vars()" can convert the object to a dict, thus we can write the data into logger file
    # for k, v in sorted(vars(configs).items()):
    #     print('\t{}: {}'.format(k, v))

#%% model

class TripletLoss(nn.Module):
    """
    https://blog.csdn.net/weixin_40671425/article/details/98068190

    Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # CDDG代码中使用mask = torch.eq(labels, labels.T).float().to(device)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

class FeatureExtractor(nn.Module):
    '''
    test code1:
    x = torch.randn((5,1,2560))
    the_model = FeatureExtractor(configs)
    fv, logits = the_model(x)
    '''
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_classes = configs.num_classes
        self.conv1 = Conv1dBlock(in_chan=1, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=32, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=16, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=5, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool5 = nn.MaxPool1d(2)

        self.flatten = nn.Flatten()


    def forward(self, x):
        x1 = self.pool1(self.conv1(x ))
        x2 = self.pool2(self.conv2(x1))
        x3 = self.pool3(self.conv3(x2))
        x4 = self.pool4(self.conv4(x3))
        x5 = self.pool5(self.conv5(x4))
        x6 = self.flatten(x5)

        return x6

class FaultClassifier(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_classes = configs.num_classes

        self.linear1 = nn.Linear(in_features=1984, out_features=128)
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(128)
        self.lrelu1  = nn.LeakyReLU()
        self.linear2 = nn.Linear(in_features=128, out_features=64)
        self.dropout2 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(64)
        self.lrelu2  = nn.LeakyReLU()
        self.linear3 = nn.Linear(in_features=64, out_features=self.num_classes)
    def forward(self, x):
        x1 = self.lrelu1(self.bn1(self.dropout1(self.linear1(x ))))
        x2 = self.lrelu2(self.bn2(self.dropout2(self.linear2(x1))))
        x3 = self.linear3(x2)

        return x3

class Discriminator(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.linear1 = nn.Linear(in_features=1984, out_features=128)
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(128)
        self.lrelu1  = nn.LeakyReLU()
        self.linear2 = nn.Linear(in_features=128, out_features=64)
        self.dropout2 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(64)
        self.lrelu2  = nn.LeakyReLU()
        self.linear3 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        x1 = self.lrelu1(self.bn1(self.dropout1(self.linear1(x ))))
        x2 = self.lrelu2(self.bn2(self.dropout2(self.linear2(x1))))
        x3 = self.linear3(x2)

        return x3


class IEDGNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        configs.num_domains = len(configs.datasets_src)
        self.configs = configs
        self.device = configs.device
        self.dataset_type = configs.dataset_type

        self.use_domain_weight = configs.use_domain_weight
        self.num_domains = configs.num_domains

        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        self.data_length = configs.data_length
        self.triplet_loss_margin = configs.triplet_loss_margin
        if self.dataset_type == 'bearing':
            self.fe = FeatureExtractor_iedg_bearing(configs).to(self.device)
            self.fc = FaultClassifier_iedg_bearing(configs).to(self.device)
            self.dc = Discriminator_iedg_bearing(configs).to(self.device)
        elif self.dataset_type == 'fan':
            self.fe = FeatureExtractor_iedg_fan(configs).to(self.device)
            self.fc = FaultClassifier_iedg_fan(configs).to(self.device)
            self.dc = Discriminator_iedg_fan(configs).to(self.device)

        self.optimizer = torch.optim.Adagrad(params=list(self.fe.parameters())+ list(self.fc.parameters())+list(self.dc.parameters()), lr = self.lr)

        # self.cross_entropy = nn.CrossEntropyLoss()
        self.cl_loss = nn.CrossEntropyLoss(reduction='none')
        self.triplet = TripletLoss(margin=self.triplet_loss_margin)

        self.lbda_t = configs.lbda_t
        self.lbda_d = configs.lbda_d

        self.weight_step = None

    def data_augument(self, x):
        '''
        x is the input signal with the shape of (B, 1, L)
        '''
        alpha_scale  = (torch.rand(1)*0.1-0.05)+1 #[0.95, 1.05]
        alpha_scale = alpha_scale.to(self.device)

        snr_db = torch.rand(1)*10.0+10.0 #[10dB, 20dB]
        snr_db = snr_db.to(self.device)

        snr = 10.0**(snr_db[0]/10.0)
        xpower = 1.0
        noise = torch.randn_like(x)*xpower/snr
        noise = noise.to(self.device)

        xn = x*alpha_scale + noise

        return xn


    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches]) # the length of the inner list is the number of the source domains (one machine is corresponding to a domain)
        labels = torch.cat([y for x, y in minibatches])
        x      = x.to(self.device)
        labels = labels.to(self.device)
        x_aug  = self.data_augument(x).to(self.device)

        fv = self.fe(x)
        fv_aug = self.fe(x_aug)
        logits = self.fc(fv)
        logits_aug = grad_reverse(self.dc(torch.cat((fv, fv_aug), dim=0)))
        labels_aug = torch.cat((torch.ones(x.shape[0]),torch.zeros(x.shape[0])), dim=0).long().to(self.device)

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

        loss_ce = torch.mean(self.cl_loss(logits, labels)*self.weight_step)

        # loss_ce = F.cross_entropy(logits, labels)

        loss_dc = F.cross_entropy(logits_aug, labels_aug)
        loss_tr = self.triplet(fv, labels)

        total_loss = loss_ce + self.lbda_d*loss_dc + self.lbda_t*loss_tr


        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        loss_ce = loss_ce.detach().cpu().data.numpy()
        loss_dc = loss_dc.detach().cpu().data.numpy()
        loss_tr = loss_tr.detach().cpu().data.numpy()


        losses={}
        losses['ce'] = loss_ce
        losses['dc'] = loss_dc
        losses['tr'] = loss_tr

        return losses

    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)

        loss_acc_result = {'loss_ce': [], 'loss_dc':[], 'loss_tr':[], 'acces':[]}

        for step in range(1, self.steps+1):
            self.train()
            self.current_step = step
            self.logger.info("================Step {}================".format(step))
            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)

            loss_acc_result['loss_ce'].append(losses['ce'])
            loss_acc_result['loss_dc'].append(losses['dc'])
            loss_acc_result['loss_tr'].append(losses['tr'])

            self.logger.info('loss_ce_train: \t {loss_ce: .4f} \t loss_dc_train: \t {loss_dc: .4f} \t loss_tr_train: \t {loss_tr: .4f}'.format(loss_ce=losses['ce'], loss_dc=losses['dc'], loss_tr=losses['tr']))

            #显示train_accuracy和test_accuracy
            if step % self.checkpoint_freq == 0 or step==self.steps:
                acc_results = self.test_model(test_loaders)
                loss_acc_result['acces'].append(acc_results)

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

    def predict(self,x):
        fv = self.fe(x)
        logits = self.fc(fv)

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

    full_path_log = os.path.join('Output//IEDGNet//log_files', datasets_list[idx])
    if not os.path.exists(full_path_log):
        os.makedirs(full_path_log)

    full_path_rep = os.path.join('Output//IEDGNet//TuneReport', datasets_list[idx])
    if not os.path.exists(full_path_rep):
        os.makedirs(full_path_rep)



    currtime = str(time.time())[:10]
    logger = create_logger(full_path_log +'//log_file'+currtime)
    # logger = create_logger('IEDGNet_hante_reproduced//log_files//log_file'+currtime)

    for i in range(1):
        model = IEDGNet(configs)
        for k, v in sorted(vars(configs).items()):
            logger.info('\t{}: {}'.format(k, v))

        loss_acc_result = model.train_model(train_minibatches_iterator, test_loaders_tgt+test_loaders_src, logger)

        loss_acc_result['loss_ce'] = np.array(loss_acc_result['loss_ce'])
        loss_acc_result['loss_dc'] = np.array(loss_acc_result['loss_dc'])
        loss_acc_result['loss_tr'] = np.array(loss_acc_result['loss_tr'])
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