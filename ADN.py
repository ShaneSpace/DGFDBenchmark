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
from models.Networks import FeatureEncoder_adn_bearing, Classifier_adn_bearing, Discriminator_adn_bearing, FeatureEncoder_adn_fan, Classifier_adn_fan, Discriminator_adn_fan
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
# from utils.AverageMeter import AverageMeter
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
# srun -w node3 --gres=gpu:1  /home/lsjia4/anaconda3/envs/pytorch/bin/python /home/lsjia4/MyFolder/fault_diagnosis/DGFDBenchmark/ADN.py


with open(os.path.join(sys.path[0], 'config_files/ADN_config.yaml')) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    print(configs) #yaml库读进来的是字典dict格式，需要用DictObj将dict转换成object
    configs = DictObj(configs)

    if configs.use_cuda and torch.cuda.is_available():
        # set(configs,'device','cuda')
        configs.device='cuda'




class DistanceMetricLoss(nn.Module):
    '''
    B = 20
    D = 1984
    x = torch.randn((B,D))
    configs.num_domains = 4
    domain_labels = torch.tensor([0]*5 +[1]*5 +[2]*5 +[3]*5)
    labels = torch.randint(low = 0, high = configs.num_classes, size=(B,))
    model = DistanceMetricLoss(configs)
    loss = model(x, labels)
    '''

    def __init__(self, configs):
        super().__init__()
        self.device = configs.device
        self.num_classes = configs.num_classes
        self.dim_feature = configs.dim_feature
        self.num_domains = configs.num_domains#len(configs.datasets_src)


    def forward(self, x, labels):
        B = x.shape[0]
        C = self.num_classes
        D = self.dim_feature
        P = self.num_domains

        # centers = []
        # distances = []
        # for i in range(C):
        #     idx_i = torch.where(labels==i,1,0) #(B,)
        #     idx_i1 = idx_i.view(idx_i.shape[0],1) #(B,1) # as a mask
        #     center_i = (idx_i1*x).sum(0, keepdim=True)/torch.sum(idx_i1) #(1,D)
        #     centers.append(center_i)

        # self.centers = torch.cat(centers, dim=0).type(torch.float32).to(self.device) #(C,D)

        # # L_intra
        # centers1 = self.prototypes.view(1,C,D)
        # centers2 = centers1.expand(B,C,D)
        # x1 = x.view(B,1,D)
        # x2 = x1.expand(B,C,D)

        # distance = (torch.abs(x2-centers2).abs().sum(2)+1e-8).pow(0.5) #(B,C)

        # L_intra
        list_category = [[] for i in range(self.num_classes)]
        for i, fv in zip(labels, x):
            fv = torch.reshape(fv, (1, fv.size(0)))
            list_category[i].append(fv)

        intra_loss = 0
        centers  = []
        for i in range(self.num_classes):
            sample_num_class_i = len(list_category[i]) #该类别下的样本数目B_i
            fv_i = torch.cat(tuple(list_category[i]), dim=0) # convert the feature vector listinto a single tensor(matrix) (B_i, D)
            center_i = torch.mean(fv_i, dim=0, keepdim=True) #(1,D)
            centers.append(center_i)
            intra_i = (fv_i - center_i).abs().sum().div(sample_num_class_i).div(C*D)   #(Bi,)
            intra_loss = intra_loss + intra_i

        # L_inter

        self.centers = torch.cat(centers, dim=0).to(self.device) # (C,D)
        center_center = self.centers.mean(dim=0, keepdim=True)   # (1,D)
        center_center1 = center_center.expand(C,D) #(C,D)
        inter_loss = (center_center1 - self.centers).abs().sum().div(C*D)

        total_loss = intra_loss - inter_loss


        return total_loss



#####################
class DGDNN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        configs.num_domains = len(configs.datasets_src)
        self.configs = configs
        self.device = configs.device
        self.dataset_type = configs.dataset_type
        self.num_domains = configs.num_domains


        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        self.use_domain_weight = configs.use_domain_weight

        self.use_learning_rate_sheduler = configs.use_learning_rate_sheduler

        if self.dataset_type== 'bearing':
            self.fe = FeatureEncoder_adn_bearing(configs).to(self.device)
            self.clf = Classifier_adn_bearing(configs).to(self.device)
            self.dcn = Discriminator_adn_bearing(configs).to(self.device)
        elif self.dataset_type== 'fan':
            self.fe = FeatureEncoder_adn_fan(configs).to(self.device)
            self.clf = Classifier_adn_fan(configs).to(self.device)
            self.dcn = Discriminator_adn_fan(configs).to(self.device)

        self.dm_loss = DistanceMetricLoss(configs)

        self.optimizer = torch.optim.Adagrad(list(self.fe.parameters()) + list(self.clf.parameters()) + list(self.dcn.parameters()), lr=self.lr)

        self.cl_loss = nn.CrossEntropyLoss(reduction='none')

        self.lbda_cl = configs.lbda_cl
        self.lbda_dc = configs.lbda_dc
        self.lbda_dm = configs.lbda_dm
        self.weight_step = None

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
        # domain labels
        domain_labels = np.repeat(np.array(list(range(self.num_domains))), self.batch_size)
        domain_labels = torch.from_numpy(domain_labels).type(torch.int64).to(self.device)

        feature_vectors = self.fe(x)
        logits = self.clf(feature_vectors)
        domain_logits = self.dcn(grad_reverse(feature_vectors))

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
        # cc_loss = self.cc_loss(feature_vectors, labels, self.weight_step)
        # ct_loss = self.ct_loss(logits, labels, self.weight_step)
        # print(logits.shape)
        # print(labels.shape)
        # print(self.weight_step.shape)
        cl_loss = torch.mean(self.cl_loss(logits, labels)*self.weight_step)
        dc_loss = F.cross_entropy(domain_logits, domain_labels)
        dm_loss = self.dm_loss(feature_vectors, labels)
        total_loss = self.lbda_cl*cl_loss  + self.lbda_dc*dc_loss + self.lbda_dm*dm_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


        loss_cl = cl_loss.detach().cpu().data.numpy()
        loss_dc = dc_loss.detach().cpu().data.numpy()
        loss_dm = dm_loss.detach().cpu().data.numpy()


        losses={}
        losses['cl'] = loss_cl
        losses['dc'] = loss_dc
        losses['dm'] = loss_dm

        return losses

    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)

        # loss_acc_result = {'loss_cc': [], 'loss_ct':[], 'loss_cl':[], 'acces':[]}
        loss_acc_result = {'loss_cl':[], 'loss_dc': [], 'loss_dm':[], 'acces':[]}

        for step in range(1, self.steps+1):
            self.train()
            self.current_step = step
            self.logger.info("================Step {}================".format(step))
            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)
            # self.scheduler.step()
            if self.use_learning_rate_sheduler:
                self.adjust_learning_rate(self.current_step)

            loss_acc_result['loss_cl'].append(losses['cl'])
            loss_acc_result['loss_dc'].append(losses['dc'])
            loss_acc_result['loss_dm'].append(losses['dm'])


            self.logger.info('loss_cl_train: \t {loss_cl: .4f} \t loss_dc_train: \t {loss_dc: .4f}  \t loss_dm_train: \t {loss_dm: .4f}'.format(loss_cl=losses['cl'], loss_dc=losses['dc'], loss_dm=losses['dm']))


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
                label_pred = self.predict(x)
                y_pred_lst.extend(label_pred.detach().cpu().data.numpy())
                y_true_lst.extend(label_fault.cpu().numpy())

            acc_i, _, _, _ = cal_index(y_true_lst, y_pred_lst) #accracy of the i-th loader
            acc_results.append(acc_i)
        self.train()

        return acc_results

    def predict(self,x):
        # print(x.shape)
        fv = self.fe(x)
        logits = self.clf(fv)

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
        configs.dim_feature = 640
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


    full_path_log = os.path.join('Output//ADN//log_files', datasets_list[idx])
    if not os.path.exists(full_path_log):
        os.makedirs(full_path_log)

    full_path_rep = os.path.join('Output//ADN//TuneReport', datasets_list[idx])
    if not os.path.exists(full_path_rep):
        os.makedirs(full_path_rep)

    currtime = str(time.time())[:10]
    logger = create_logger(full_path_log +'//log_file'+currtime)
    # logger = create_logger('IDEA_test//log_files//log_file'+currtime)
    for i in range(1):
        model = DGDNN(configs)
        for k, v in sorted(vars(configs).items()):
            logger.info('\t{}: {}'.format(k, v))

        loss_acc_result = model.train_model(train_minibatches_iterator, test_loaders_tgt+test_loaders_src, logger)

        loss_acc_result['loss_cl'] = np.array(loss_acc_result['loss_cl'])
        loss_acc_result['loss_dc'] = np.array(loss_acc_result['loss_dc'])
        loss_acc_result['loss_dm'] = np.array(loss_acc_result['loss_dm'])
        loss_acc_result['acces']   = np.array(loss_acc_result['acces'])


        # # save the loss curve and acc curve

        sio.savemat(full_path_log+'//loss_acc_result'+currtime+'.mat',loss_acc_result)
        gen_report = GenReport(full_path_rep)
        gen_report.write_file(configs=configs, test_item=None, loss_acc_result=loss_acc_result)
        gen_report.save_file(currtime)

        # update current time
        currtime = str(time.time())[:10]



if __name__ == '__main__':
    main(0, configs)
    # for idx in range(5):
    #     main(idx, configs)



# mfs_data = read_mfs.read_data_file()
# mfs_dataset_train = SimpleDataset(mfs_data['train'])
# batch_data = mfs_dataset_train[:5]
# x_out = the_model.forward(*batch_data)



