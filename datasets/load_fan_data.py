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



import scipy.io as sio



from utils.DatasetClass import InfiniteDataLoader, SimpleDataset




class ReadMIMII():
    def __init__(self, domain, section, configs):
        self.configs = configs
        self.dataset_debug = configs.dataset_debug
        self.section = section
        self.domain = domain
        if self.section=='00':
            self.domains = ['W','X','Y','Z']
        elif self.section=='01':
            self.domains=['A','B','C']
        else:
            self.domains=['L1','L2','L3','L4']
        self.batch_size = configs.batch_size

    def read_data_file(self):

        if self.dataset_debug:
            if self.section=='00':
                datafile_path = '../../Data/MIMIIDG/mimii_fan_sec00'
            elif self.section=='01':
                datafile_path = '../../Data/MIMIIDG/mimii_fan_sec01'
            else:
                datafile_path = '../../Data/MIMIIDG/mimii_fan_sec02'
        else:
            if self.section=='00':
                datafile_path = '../Data/MIMIIDG/mimii_fan_sec00'
            elif self.section=='01':
                datafile_path = '../Data/MIMIIDG/mimii_fan_sec01'
            else:
                datafile_path = '../Data/MIMIIDG/mimii_fan_sec02'

        data = sio.loadmat(datafile_path)
        data_domain = data[self.domain]
        data_domain_data = torch.from_numpy(data_domain['data'][0,0]).unsqueeze(dim=1).type(torch.float32)
        data_domain_label = torch.from_numpy(data_domain['label'][0,0]).squeeze().type(torch.int64)
        data_domain_torch = {}
        data_domain_torch['data'] = data_domain_data
        data_domain_torch['label'] = data_domain_label

        the_data = {}
        the_data['train'] = data_domain_torch
        the_data['test']  = data_domain_torch

        return the_data

    def load_dataloaders(self):
        the_data = self.read_data_file()

        dataset_train = SimpleDataset(the_data['train'])
        dataset_test  = SimpleDataset(the_data['test'] )

        dataloader_params_test = dict(batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=0,  # use main thread only or may receive multiple batches
                        drop_last=False,
                        pin_memory=False)

        train_loader = InfiniteDataLoader(dataset=dataset_train, batch_size=self.batch_size)
        test_loader  = DataLoader(dataset_test,**dataloader_params_test)
        return train_loader, test_loader

