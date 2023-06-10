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
from scipy.signal import butter, lfilter, freqz, hilbert
from scipy.interpolate import interp1d
import pickle
import matplotlib.pyplot as plt
import math

from models.Conv1dBlock import Conv1dBlock
from models.Networks import FeatureGenerator_bearing, FaultClassifier_bearing, DomainClassifier_bearing, FeatureGenerator_fan, FaultClassifier_fan, DomainClassifier_fan
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
# srun -w node3 --gres=gpu:1  /home/lsjia4/anaconda3/envs/pytorch/bin/python /home/lsjia4/MyFolder/fault_diagnosis/DGFDBenchmark/DGNIS.py




with open(os.path.join(sys.path[0], 'config_files/DGNIS_config.yaml')) as f:
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


class CoralLoss(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.batch_size = configs.batch_size
        self.num_domains = len(configs.datasets_src)

    def forward(self, source, target):
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)

        # frobenius norm between source and target
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4*d*d)
        return loss

    def cal_overall_coral_loss(self, features):
        '''
        Args:
            features, should be a list with feature vectors from different domains
        '''
        loss = 0
        for i in range(self.num_domains):
            for j in range(i+1, self.num_domains):
                loss += self.forward(features[i*self.batch_size:(i+1)*self.batch_size],
                features[j*self.batch_size:(j+1)*self.batch_size])
                # loss += self.forward(features[i], features[j])

        return loss


class DGNIS(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.dataset_type =  configs.dataset_type

        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size =  configs.batch_size
        self.margin = configs.margin

        self.use_domain_weight = configs.use_domain_weight
        self.domain_weight_scale = configs.domain_weight_scale

        self.num_domains = len(configs.datasets_src)
        if self.dataset_type == 'bearing':
            self.fe_inv = FeatureGenerator_bearing(configs).to(self.device)
            self.fe_dom = FeatureGenerator_bearing(configs).to(self.device)
            self.dc = DomainClassifier_bearing(configs).to(self.device)
            self.fc1 = FaultClassifier_bearing(configs).to(self.device)
            self.fc2 = FaultClassifier_bearing(configs).to(self.device)
            self.fc3 = FaultClassifier_bearing(configs).to(self.device)
            self.fc4 = FaultClassifier_bearing(configs).to(self.device)
            self.fc5 = FaultClassifier_bearing(configs).to(self.device)
            self.fc6 = FaultClassifier_bearing(configs).to(self.device)
            self.fc7 = FaultClassifier_bearing(configs).to(self.device)
            self.fc8 = FaultClassifier_bearing(configs).to(self.device)
        elif self.dataset_type == 'fan':
            self.fe_inv = FeatureGenerator_fan(configs).to(self.device)
            self.fe_dom = FeatureGenerator_fan(configs).to(self.device)
            self.dc = DomainClassifier_fan(configs).to(self.device)
            self.fc1 = FaultClassifier_fan(configs).to(self.device)
            self.fc2 = FaultClassifier_fan(configs).to(self.device)
            self.fc3 = FaultClassifier_fan(configs).to(self.device)
            self.fc4 = FaultClassifier_fan(configs).to(self.device)
            self.fc5 = FaultClassifier_fan(configs).to(self.device)
            self.fc6 = FaultClassifier_fan(configs).to(self.device)
            self.fc7 = FaultClassifier_fan(configs).to(self.device)
            self.fc8 = FaultClassifier_fan(configs).to(self.device)

        self.fcs = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6, self.fc7, self.fc8]
        self.fcs = self.fcs[:self.num_domains]

        self.coral_loss = CoralLoss(configs)
        self.triplet_loss =  TripletLoss(margin=self.margin)

        self.optimizer = torch.optim.Adam(params=list(self.parameters()), lr = self.lr)
        self.lbda_cr = configs.lbda_cr
        self.lbda_tp = configs.lbda_tp

        lr_list = [0.0001/(1+10*p/self.steps)**0.75 for p in range(self.steps+1)]
        lambda_para = lambda step: lr_list[step]
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_para)

        self.weight_step = None


    def update(self, minibatches):
        self.train()
        # x = [x.to(self.device) for x, y in minibatches] # the length of the inner list is the number of the source domains (one machine is corresponding to a domain)
        # labels = [y.to(self.device) for x, y in minibatches]
        x = torch.cat([x for x, y in minibatches]) # the length of the inner list is the number of the source domains (one machine is corresponding to a domain)
        labels = torch.cat([y for x, y in minibatches])
        x      = x.to(self.device)
        labels = labels.to(self.device)

        fv_inv = self.fe_inv(x)

        # calculate the cross entropy of each domain and add them together
        loss_ce_total = 0
        loss_ce_list = []
        for i in range(self.num_domains):
            fv_inv_i = fv_inv[i*self.batch_size:(i+1)*self.batch_size]
            labels_i = labels[i*self.batch_size:(i+1)*self.batch_size]
            logits_i = self.fcs[i](fv_inv_i)
            cross_entropy_i = F.cross_entropy(logits_i, labels_i)
            loss_ce_list.append(cross_entropy_i)
            loss_ce_total += cross_entropy_i


        if self.use_domain_weight:
            if  self.weight_step is None:
                self.weight_step = torch.ones(x.shape[0]).to(self.device)
            else:
                ce_value_domains =  torch.tensor(loss_ce_list).to(self.device)
                weight_step = 1 + ce_value_domains/loss_ce_total
                self.weight_step =  weight_step.to(self.device)

            loss_ce =0
            for i in range(self.num_domains):
                loss_ce +=  self.weight_step[i]*loss_ce_list[i]
        else:
            loss_ce = loss_ce_total


        # calculate the coral loss
        loss_cr = self.coral_loss.cal_overall_coral_loss(fv_inv)
        # calculate the triplet loss
        loss_tp = self.triplet_loss(fv_inv, labels)
        # calculate the domain classification loss
        dom_labels = torch.arange(0, self.num_domains).repeat(self.batch_size,1).T.flatten(0).long().to(self.device)
        fv_dom = self.fe_dom(x)
        dom_logits  =self.dc(fv_dom)
        loss_cd = F.cross_entropy(dom_logits, dom_labels)

        total_loss = loss_ce + self.lbda_cr*loss_cr + self.lbda_tp*loss_tp + loss_cd

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        loss={}
        loss['ce'] = loss_ce.detach().cpu().data.numpy()
        loss['cr'] = loss_cr.detach().cpu().data.numpy()
        loss['tp'] = loss_tp.detach().cpu().data.numpy()
        loss['cd'] = loss_cd.detach().cpu().data.numpy()

        return loss

    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)

        loss_acc_result = {'loss_ce': [], 'loss_cr':[], 'loss_tp':[], 'loss_cd':[], 'acces':[]}

        for step in range(0, self.steps):
            self.train()
            self.current_step = step
            self.logger.info("================Step {}================".format(step+1))
            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)

            loss_acc_result['loss_ce'].append(losses['ce'])
            loss_acc_result['loss_cr'].append(losses['cr'])
            loss_acc_result['loss_tp'].append(losses['tp'])
            loss_acc_result['loss_cd'].append(losses['cd'])

            self.logger.info('loss_ce_train: \t {loss_ce: .4f} \t loss_cr_train: \t {loss_cr: .4f} \t loss_tp_train: \t {loss_tp: .4f} \t loss_cd_train: \t {loss_cd: .4f}'.format(loss_ce=losses['ce'], loss_cr=losses['cr'], loss_tp=losses['tp'], loss_cd=losses['cd']))

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


    def predict(self, x):
        self.eval()

        fv_inv = self.fe_inv(x)
        logits=[]
        for fc_i in self.fcs:
            logits.append(fc_i(fv_inv).detach().cpu().data.numpy())

        logits = torch.from_numpy(np.array(logits)).to(self.device)

        # logits = torch.tensor(logits).to(self.device) #(num_domain, num_sample, num_classes)
        logits = torch.permute(logits, [1,0,2]) #(num_sample, num_domain, num_classes)

        fv_dom = self.fe_dom(x)
        logits_dc  = self.dc(fv_dom)
        w = F.softmax(logits_dc, dim=1) #(num_sample, num_domain)
        w = torch.unsqueeze(w, dim=2)
        # print(w.shape)
        # print(logits.shape)

        logits_w = logits * w #(num_sample, num_domain, num_classes)
        logits_w_sum = torch.sum(logits_w, dim=1) #(num_sample, num_classes)

        return torch.max(logits_w_sum, dim=1)[1]






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

    full_path_log = os.path.join('Output//DGNIS//log_files', datasets_list[idx])
    if not os.path.exists(full_path_log):
        os.makedirs(full_path_log)

    full_path_rep = os.path.join('Output//DGNIS//TuneReport', datasets_list[idx])
    if not os.path.exists(full_path_rep):
        os.makedirs(full_path_rep)


    currtime = str(time.time())[:10]
    logger = create_logger(full_path_log +'//log_file'+currtime)

    for i in range(1):
        model = DGNIS(configs)
        for k, v in sorted(vars(configs).items()):
            logger.info('\t{}: {}'.format(k, v))

        loss_acc_result = model.train_model(train_minibatches_iterator, test_loaders_tgt+test_loaders_src, logger)



        loss_acc_result['loss_ce'] = np.array(loss_acc_result['loss_ce'])
        loss_acc_result['loss_cr'] = np.array(loss_acc_result['loss_cr'])
        loss_acc_result['loss_tp'] = np.array(loss_acc_result['loss_tp'])
        loss_acc_result['loss_cd'] = np.array(loss_acc_result['loss_cd'])
        loss_acc_result['acces']   = np.array(loss_acc_result['acces'])


        # # save the loss curve and acc curve
        sio.savemat(full_path_log+'//loss_acc_result'+currtime+'.mat',loss_acc_result)
        gen_report = GenReport(full_path_rep)
        gen_report.write_file(configs=configs, test_item=None, loss_acc_result=loss_acc_result)
        gen_report.save_file(currtime)


        # torch.save(model.to('cpu'),'Output//DGNIS//saved_models//model'+currtime+'.pt')
        currtime = str(time.time())[:10]



if __name__ == '__main__':
    main(1, configs)
    # for i in range(5):
    #     main(i, configs)



# mfs_data = read_mfs.read_data_file()
# mfs_dataset_train = SimpleDataset(mfs_data['train'])
# batch_data = mfs_dataset_train[:5]
# x_out = the_model.forward(*batch_data)




# %%