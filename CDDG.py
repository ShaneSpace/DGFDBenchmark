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
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt
import math

from models.Conv1dBlock import Conv1dBlock
from models.Networks import Encoder_cddg_bearing, Decoder_cddg_bearing,Classifier_cddg_bearing,Encoder_cddg_fan, Decoder_cddg_fan,Classifier_cddg_fan
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

# run code
# srun -w node3 --gres=gpu:1  /home/lsjia4/anaconda3/envs/pytorch/bin/python /home/lsjia4/MyFolder/fault_diagnosis/DGFDBenchmark/CDDG.py


with open(os.path.join(sys.path[0], 'config_files/CDDG_config.yaml')) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    print(configs)
    configs = DictObj(configs)

    if configs.use_cuda and torch.cuda.is_available():
        configs.device='cuda'

class CDDG(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs =configs
        self.device = configs.device
        self.dataset_type = configs.dataset_type

        self.num_classes = configs.num_classes
        self.batch_size = configs.batch_size
        self.steps = configs.steps # 参数的更新次数，默认200
        self.checkpoint_freq = configs.checkpoint_freq # 每更新checkpoint_freq个batch,对test dataloader进行推理1次,默认100
        self.lr = configs.lr
        self.num_domains = len(configs.datasets_src)

        if self.dataset_type=='bearing':
            self.encoder_m = Encoder_cddg_bearing()
            self.encoder_h = Encoder_cddg_bearing()
            self.decoder   = Decoder_cddg_bearing()
            self.classifer = Classifier_cddg_bearing(self.num_classes)
        elif self.dataset_type == 'fan':
            self.encoder_m = Encoder_cddg_fan()
            self.encoder_h = Encoder_cddg_fan()
            self.decoder   = Decoder_cddg_fan()
            self.classifer = Classifier_cddg_fan(self.num_classes)

        self.optimizer =  torch.optim.Adam(list(self.encoder_m.parameters())+ list(self.encoder_h.parameters()) + list(self.decoder.parameters()) +list(self.classifer.parameters()) , lr=self.lr)

        self.w_rc = configs.w_rc
        self.w_rr = configs.w_rr
        self.w_ca = configs.w_ca

        self.weight_step = None
        self.use_domain_weight = configs.use_domain_weight

        self.use_learning_rate_sheduler = configs.use_learning_rate_sheduler
        self.gamma = configs.gamma

    def forward_penul_fv(self, x):
        _, fh_vec = self.encoder_h(x) #fh_vec:(B,D)
        fv = self.classifer.forward1(fh_vec)

        return fv

    def forward_zd_fv(self, x):
        _, fm_vec = self.encoder_m(x) #fh_vec:(B,D)

        return fm_vec

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
                lr *= self.gamma if step >= milestone else 1.0
        for param_group in self.optimizer.param_groups:
            # print(lr)
            param_group["lr"] = lr

    def cal_reconstruction_loss(self, x, x_rec):
        # return (x_rec-x).pow(2).sum()/x.shape[0]
        return (x_rec-x).pow(2).mean()

    def cal_reduce_redundancy_loss(self, fm_vec, fh_vec):
        '''
        zz = torch.load('fm_fh_tensor.pt',map_location=torch.device('cpu') )
        fm_vec = zz[0]
        fh_vec = zz[1]
        '''
        lbd = 1
        B = fm_vec.shape[0]
        D = fm_vec.shape[1]
        # debug
        # torch.save([fm_vec, fh_vec],'fm_fh_tensor.pt')
        #注意这里，原来是在dim=1上标准化，这个是错误的，这里一列才是一个vector，所以应该是dim=0标准化
        fm_vec = F.normalize(fm_vec, p=2, dim=0) #(B,D)
        fh_vec = F.normalize(fh_vec, p=2, dim=0) #(B,D)
        sim_fm_vec = torch.matmul(fm_vec.T, fm_vec) #(D,D)
        sim_fh_vec = torch.matmul(fh_vec.T, fh_vec) #(D,D)
        # 经过normalize之后，上边两个矩阵的对角线本身就是1（不同于Barlow Twins, 这里是两个相同向量的内积）

        E = torch.eye(D).to(self.device)

        loss_fm =  ((1-E)*sim_fm_vec).pow(2).sum()/torch.sum(1-E)
        loss_fh =  ((1-E)*sim_fh_vec).pow(2).sum()/torch.sum(1-E)

        loss_fmh = torch.matmul(fh_vec.T, fm_vec).div(B).pow(2).mean()
        # loss = loss_fmh

        loss = loss_fm + loss_fh + loss_fmh
        # loss =   loss_fm + loss_fh


        return loss

    def cal_causal_aggregation_loss(self, fm_vec, fh_vec, labels, domain_labels):
        B = fm_vec.shape[0]
        D = fm_vec.shape[1]

        fm_vec = F.normalize(fm_vec, p=2, dim=1) # (B,D)
        fh_vec = F.normalize(fh_vec, p=2, dim=1) # (B,D)

        labels= labels.contiguous().view(-1, 1)
        mask_fh = torch.eq(labels, labels.T).float().to(self.device) # (B,B)
        sim_fh_vec = torch.matmul(fh_vec, fh_vec.T)/D # (B,B)
        loss_fh = -(mask_fh*sim_fh_vec).sum()/torch.sum(mask_fh) + ((1-mask_fh)*sim_fh_vec).sum()/torch.sum(1-mask_fh)
        # loss_fh = -(mask_fh*sim_fh_vec).sum() + ((1-mask_fh)*sim_fh_vec).sum() #似乎会带来不稳定（在MFS表现上）

        domain_labels= domain_labels.contiguous().view(-1, 1)
        mask_fm = torch.eq(domain_labels, domain_labels.T).float().to(self.device) # (B,B)
        sim_fm_vec = torch.matmul(fm_vec, fm_vec.T)/D # (B,B)
        loss_fm = -(mask_fm*sim_fm_vec).sum()/torch.sum(mask_fm) + ((1-mask_fm)*sim_fm_vec).sum()/torch.sum(1-mask_fm)
        # loss_fm = -(mask_fm*sim_fm_vec).sum() + ((1-mask_fm)*sim_fm_vec).sum()

        loss = loss_fm + loss_fh

        return loss


    def forward(self, x, labels):
        output = {}
        B = x.shape[0] # total_batch
        domain_labels = np.repeat(np.array(list(range(self.num_domains))), self.batch_size)
        domain_labels = torch.from_numpy(domain_labels).type(torch.int64).to(self.device)

        fm_map, fm_vec = self.encoder_m(x) #fm_vec:(B,D)
        fh_map, fh_vec = self.encoder_h(x) #fh_vec:(B,D)

        fmh_map = torch.cat([fm_map, fh_map], dim=1) #(B,C,L)(B,C,L)->(B,2C,L)
        x_rec = self.decoder(fmh_map)
        logits = self.classifer(fh_vec)

        loss_rc = self.cal_reconstruction_loss(x, x_rec)
        loss_rr = self.cal_reduce_redundancy_loss(fm_vec, fh_vec)
        loss_ca = self.cal_causal_aggregation_loss(fm_vec, fh_vec, labels, domain_labels)
        loss_cl = F.cross_entropy(logits, labels)

        if self.use_domain_weight:
            if  self.weight_step is None:
                self.weight_step = torch.ones(B).to(self.device)
            else:
                ce_values  = F.cross_entropy(logits, labels, reduction='none')
                ce_values_2d = torch.reshape(ce_values, (self.num_domains, self.batch_size))
                ce_value_domain = torch.mean(ce_values_2d,dim=1)
                ce_value_sum = torch.sum(ce_value_domain)
                weight_step = 1 + ce_value_domain/ce_value_sum
                self.weight_step = weight_step.repeat((self.batch_size,1)).T.flatten(0).to(self.device)
        else:
            self.weight_step = torch.ones(B).to(self.device)

        loss_cl = torch.mean(F.cross_entropy(logits, labels, reduction='none')*self.weight_step)


        output['loss_rc'] = loss_rc
        output['loss_rr'] = loss_rr
        output['loss_ca'] = loss_ca
        output['loss_cl'] = loss_cl

        return output

    def update(self, minibatches):

        x = torch.cat([x for x, y in minibatches])
        labels_fault = torch.cat([y for x, y in minibatches])

        # x, labels_fault = minibatches
        x, labels_fault =  x.to(self.device), labels_fault.to(self.device)
        output = self.forward(x, labels_fault)

        loss_rc = output['loss_rc']
        loss_rr = output['loss_rr']
        loss_ca = output['loss_ca']
        loss_cl = output['loss_cl']

        loss = self.w_rc*loss_rc + self.w_rr*loss_rr + self.w_ca*loss_ca + loss_cl
        # loss = loss_rc + loss_ca + loss_cl

        # print('loss_rc', loss_rc)
        # print('loss_rr', loss_rr)
        # print('loss_ca', loss_ca)
        # print('loss_cl', loss_cl)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        losses = {}
        losses['rc'] = loss_rc.detach().cpu().data.numpy()
        losses['rr'] = loss_rr.detach().cpu().data.numpy()
        losses['ca'] = loss_ca.detach().cpu().data.numpy()
        losses['cl'] = loss_cl.detach().cpu().data.numpy()

        return losses

    def train_model(self, train_minibatches_iterator, test_loaders, logger):
        self.logger = logger
        self.to(self.device)

        # loss_acc_result = {'loss_cc': [], 'loss_ct':[], 'loss_cl':[], 'acces':[]}
        loss_acc_result = {'loss_rc': [], 'loss_rr': [], 'loss_ca': [], 'loss_cl':[], 'acces':[]}

        for step in range(1, self.steps+1):
            self.train()
            self.current_step = step
            self.logger.info("================Step {}================".format(step))
            minibatches_device = next(train_minibatches_iterator)
            losses = self.update(minibatches_device)
            # self.scheduler.step()
            if self.use_learning_rate_sheduler:
                self.adjust_learning_rate(self.current_step)

            loss_acc_result['loss_rc'].append(losses['rc'])
            loss_acc_result['loss_rr'].append(losses['rr'])
            loss_acc_result['loss_ca'].append(losses['ca'])
            loss_acc_result['loss_cl'].append(losses['cl'])


            # self.logger.info('loss_cl_train: \t {loss_cl: .4f} \t loss_cc_train: \t {loss_cc: .4f} \t loss_ct_train: \t {loss_ct: .4f} \t loss_cl_train: \t {loss_cl: .4f}'.format(loss_cl=losses['cl'], loss_cc=losses['cc'], loss_ct=losses['ct']))
            self.logger.info('loss_rc_train: \t {loss_rc: .4f} \t loss_rr_train: \t {loss_rr: .4f} \t loss_ca_train: \t {loss_ca: .4f} \t loss_cl_train: \t {loss_cl: .4f}'.format(loss_rc=losses['rc'], loss_rr=losses['rr'] ,loss_ca=losses['ca'], loss_cl=losses['cl']))


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
        confusion_matrix_results = []
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

            if self.current_step == self.steps:
                cm_i = confusion_matrix(y_true_lst, y_pred_lst)
                confusion_matrix_results.append(cm_i)
                print(cm_i)

            acc_i, _, _, _ = cal_index(y_true_lst, y_pred_lst) #accracy of the i-th loader
            acc_results.append(acc_i)
        self.train()
        # print('############debug')
        # print(acc_results)

        return acc_results

    def predict(self, x):
        '''
        预测样本的标签
        '''
        _, fh_vec= self.encoder_h(x)
        y_pred = self.classifer(fh_vec)

        return torch.max(y_pred, dim=1)[1]



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

    full_path_log = os.path.join('Output//CDDG//log_files', datasets_list[idx])
    if not os.path.exists(full_path_log):
        os.makedirs(full_path_log)

    full_path_rep = os.path.join('Output//CDDG//TuneReport', datasets_list[idx])
    if not os.path.exists(full_path_rep):
        os.makedirs(full_path_rep)


    currtime = str(time.time())[:10]
    logger = create_logger(full_path_log +'//log_file'+currtime)
    for i in range(1):
        model = CDDG(configs)
        for k, v in sorted(vars(configs).items()):
            logger.info('\t{}: {}'.format(k, v))

        loss_acc_result = model.train_model(train_minibatches_iterator, test_loaders_tgt+test_loaders_src, logger)


        loss_acc_result['loss_rc'] = np.array(loss_acc_result['loss_rc'])
        loss_acc_result['loss_rr'] = np.array(loss_acc_result['loss_rr'])
        loss_acc_result['loss_ca'] = np.array(loss_acc_result['loss_ca'])
        loss_acc_result['loss_cl'] = np.array(loss_acc_result['loss_cl'])
        loss_acc_result['acces'] = np.array(loss_acc_result['acces'])


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