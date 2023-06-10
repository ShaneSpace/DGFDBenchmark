import torch
import torch.nn as nn
from models.Conv1dBlock import Conv1dBlock


#########################################
class Network_bearing(nn.Module):
    '''
    test code1:
    x = torch.randn((5,1,2560))
    the_model = Network(configs)
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
        self.linear1 = nn.Linear(in_features=1984, out_features=300)
        self.lrelu1  = nn.LeakyReLU()
        self.linear2 = nn.Linear(in_features=300, out_features=self.num_classes)

    def forward(self, x):
        x1 = self.pool1(self.conv1(x ))
        x2 = self.pool2(self.conv2(x1))
        x3 = self.pool3(self.conv3(x2))
        x4 = self.pool4(self.conv4(x3))
        x5 = self.pool5(self.conv5(x4))
        x6 = self.flatten(x5)
        x7 = self.linear1(x6)
        x8 = self.linear2(x7)

        return x6, x8 # x6 is the feature vector Z, x8 is the output logits for classification

    def forward_penul_fv(self, x):
        x1 = self.pool1(self.conv1(x ))
        x2 = self.pool2(self.conv2(x1))
        x3 = self.pool3(self.conv3(x2))
        x4 = self.pool4(self.conv4(x3))
        x5 = self.pool5(self.conv5(x4))
        x6 = self.flatten(x5)

        x7 = self.linear1(x6)

        return x7

###################################
class Network_fan(nn.Module):
    '''
    test code1:
    x = torch.randn((5,1,20032))
    the_model = Network(configs)
    fv, logits = the_model(x)
    '''
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_classes = configs.num_classes
        self.conv1 = Conv1dBlock(in_chan=1, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool5 = nn.MaxPool1d(2)

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.3)
        self.linear1 = nn.Linear(in_features=640, out_features=300)
        self.lrelu1  = nn.LeakyReLU()
        self.linear2 = nn.Linear(in_features=300, out_features=self.num_classes)

    def forward(self, x):
        x1 = self.pool1(self.conv1(x ))
        x2 = self.pool2(self.conv2(x1))
        x3 = self.pool3(self.conv3(x2))
        x4 = self.pool4(self.conv4(x3))
        x5 = self.pool5(self.conv5(x4))
        x6 = self.flatten(x5)
        x7 = self.linear1(x6)
        x8 = self.linear2(x7)

        return x6, x8 # x6 is the feature vector Z, x8 is the output logits for classification

    def forward_penul_fv(self, x):
        x1 = self.pool1(self.conv1(x ))
        x2 = self.pool2(self.conv2(x1))
        x3 = self.pool3(self.conv3(x2))
        x4 = self.pool4(self.conv4(x3))
        x5 = self.pool5(self.conv5(x4))
        x6 = self.flatten(x5)
        x7 = self.linear1(x6)

        return x7
###########
###########
class FeatureGenerator_bearing(nn.Module):
    '''
    DGNIS
    test code:
    the_model = FeatureGenerator()
    x = torch.randn((5,1,1024))
    y = the_model(x)
    print(y.shape)
    torch.Size([5, 3520])
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

class FaultClassifier_bearing(nn.Module):
    '''DGNIS'''
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_classes = configs.num_classes

        self.linear1 = nn.Linear(in_features = 1984, out_features = 300)
        self.linear2 = nn.Linear(in_features = 300, out_features = self.num_classes)

    def forward(self,x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)

        return x2

class DomainClassifier_bearing(nn.Module):
    '''DGNIS'''
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_domains = len(configs.datasets_src)

        self.linear1 = nn.Linear(in_features = 1984, out_features = 300)
        self.linear2 = nn.Linear(in_features = 300, out_features =  self.num_domains)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)

        return x2


class FeatureGenerator_fan(nn.Module):
    '''
    DGNIS
    test code:
    the_model = FeatureGenerator()
    x = torch.randn((5,1,1024))
    y = the_model(x)
    print(y.shape)
    torch.Size([5, 3520])
    '''
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_classes = configs.num_classes
        self.conv1 = Conv1dBlock(in_chan=1, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
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

class FaultClassifier_fan(nn.Module):
    '''DGNIS'''
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_classes = configs.num_classes

        self.linear1 = nn.Linear(in_features=640, out_features=300)
        self.linear2 = nn.Linear(in_features=300, out_features=self.num_classes)

    def forward(self,x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        return x2

class DomainClassifier_fan(nn.Module):
    '''DGNIS'''
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_domains = len(configs.datasets_src)

        self.linear1 = nn.Linear(in_features = 640, out_features = 300)
        self.linear2 = nn.Linear(in_features = 300, out_features =  self.num_domains)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)

        return x2


################################################
class FeatureExtractor_iedg_bearing(nn.Module):
    '''
    IEDGNet
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

class FaultClassifier_iedg_bearing(nn.Module):
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

class Discriminator_iedg_bearing(nn.Module):
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

class FeatureExtractor_iedg_fan(nn.Module):
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
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
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

class FaultClassifier_iedg_fan(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_classes = configs.num_classes

        self.linear1 = nn.Linear(in_features=640, out_features=128)
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


class Discriminator_iedg_fan(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.linear1 = nn.Linear(in_features=640, out_features=128)
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


##############
class FeatureEncoder_adn_bearing(nn.Module):
    def __init__(self, configs):
        super().__init__()
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

class Classifier_adn_bearing(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_classes = configs.num_classes

        self.linear1 = nn.Linear(in_features=1984, out_features=300)
        self.lrelu1  = nn.LeakyReLU()
        self.linear2 = nn.Linear(in_features=300, out_features=self.num_classes)

    def forward(self, x):
        x1 = self.lrelu1(self.linear1(x))
        x2 = self.linear2(x1)

        return x2

class Discriminator_adn_bearing(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_domains = len(configs.datasets_src)

        self.linear1 = nn.Linear(in_features=1984, out_features=300)
        self.lrelu1  = nn.LeakyReLU()
        self.linear2 = nn.Linear(in_features=300, out_features=self.num_domains)

    def forward(self, x):
        x1 = self.lrelu1(self.linear1(x))
        x2 = self.linear2(x1)

        return x2


class FeatureEncoder_adn_fan(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_classes = configs.num_classes
        self.conv1 = Conv1dBlock(in_chan=1, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=0)
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

class Classifier_adn_fan(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_classes = configs.num_classes

        self.linear1 = nn.Linear(in_features=640, out_features=300)
        self.lrelu1  = nn.LeakyReLU()
        self.linear2 = nn.Linear(in_features=300, out_features=self.num_classes)

    def forward(self, x):
        x1 = self.lrelu1(self.linear1(x))
        x2 = self.linear2(x1)

        return x2

class Discriminator_adn_fan(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_domains = len(configs.datasets_src)

        self.linear1 = nn.Linear(in_features=640, out_features=300)
        self.lrelu1  = nn.LeakyReLU()
        self.linear2 = nn.Linear(in_features=300, out_features=self.num_domains)

    def forward(self, x):
        x1 = self.lrelu1(self.linear1(x))
        x2 = self.linear2(x1)

        return x2

