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

#############
class Encoder_cddg_bearing(nn.Module):
    def __init__(self):
        super().__init__()
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
        f_map = self.pool5(self.conv5(x4))
        f_vec = self.flatten(f_map)
        # print(x1.shape, x2.shape, x3.shape, x4.shape, f_map.shape)

        return f_map, f_vec # f_map: feature maps (B,C,L); f_vec: feature vector (B, C*L)

class Decoder_cddg_bearing(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode ='linear', align_corners=True)
        self.conv1 = Conv1dBlock(in_chan=32*2, out_chan=32, kernel_size=5, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=4)
        self.up2 = nn.Upsample(scale_factor=2, mode ='linear', align_corners=True)
        self.conv2 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=15, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=15)
        self.up3 = nn.Upsample(scale_factor=2, mode ='linear', align_corners=True)
        self.conv3 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=31, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=31)
        self.up4 = nn.Upsample(scale_factor=2, mode ='linear', align_corners=True)
        self.conv4 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=63, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=63)
        self.up5 = nn.Upsample(scale_factor=2, mode ='linear', align_corners=True)
        self.conv5 = Conv1dBlock(in_chan=32, out_chan=32,  kernel_size=127, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=127)
        #最后一层输出不加Activation
        self.conv6 = Conv1dBlock(in_chan=32, out_chan=1,  kernel_size=127, stride=1, activation = 'none', norm='BN', pad_type='reflect', padding=63)


    def forward(self, x):
        '''
        'x' is the feature maps
        '''
        x1 = self.conv1(self.up1(x ))
        x2 = self.conv2(self.up2(x1))
        x3 = self.conv3(self.up3(x2))
        x4 = self.conv4(self.up4(x3))
        x5 = self.conv5(self.up5(x4))
        # print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        # >> torch.Size([5, 32, 128]) torch.Size([5, 32, 272]) torch.Size([5, 32, 576]) torch.Size([5, 32, 1216]) torch.Size([5, 32, 2560])

        x_rec = self.conv6(x5)

        return x_rec

class Classifier_cddg_bearing(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(in_features=1984, out_features=300)
        self.linear2 = nn.Linear(in_features=300, out_features=3)
    def forward(self, x):
        '''
        'x' is the feature vector
        '''
        x1 = self.linear1(x)
        x2 = self.linear2(x1)

        return x2  # logits
    def forward1(self,x):
        x1 = self.linear1(x)
        return x1

class Encoder_cddg_fan(nn.Module):
    def __init__(self):
        super().__init__()

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
        f_map = self.pool5(self.conv5(x4)) # (B, 32, 20)
        f_vec = self.flatten(f_map) # (B, 640)
        # print(x1.shape, x2.shape, x3.shape, x4.shape, f_map.shape)

        return f_map, f_vec # f_map: feature maps (B,C,L); f_vec: feature vector (B, C*L)


class Decoder_cddg_fan(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode ='linear', align_corners=True) #(B,64, 40)
        self.conv1 = Conv1dBlock(in_chan=32*2, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='zero', padding=63) #(B,32,103)
        self.up2 = nn.Upsample(scale_factor=2, mode ='linear', align_corners=True) #(B,32,206)
        self.conv2 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=64, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=64) #(B,32, 271)
        self.up3 = nn.Upsample(scale_factor=4, mode ='linear', align_corners=True) #(B,32,1084)
        self.conv3 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=127) #(B,32,1211)
        self.up4 = nn.Upsample(scale_factor=4, mode ='linear', align_corners=True) #(B,32,4844)
        self.conv4 = Conv1dBlock(in_chan=32, out_chan=32, kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=129) #(B,32,4975)
        self.up5 = nn.Upsample(scale_factor=4, mode ='linear', align_corners=True) #(B,32,19900)
        self.conv5 = Conv1dBlock(in_chan=32, out_chan=32,  kernel_size=128, stride=1, activation = 'lrelu', norm='BN', pad_type='reflect', padding=129)#(B,32,20031)
        #最后一层输出不加Activation
        self.conv6 = Conv1dBlock(in_chan=32, out_chan=1,  kernel_size=128, stride=1, activation = 'none', norm='BN', pad_type='reflect', padding=64)


    def forward(self, x):
        '''
        'x' is the feature maps
        '''
        x1 = self.conv1(self.up1(x ))
        x2 = self.conv2(self.up2(x1))
        x3 = self.conv3(self.up3(x2))
        x4 = self.conv4(self.up4(x3))
        x5 = self.conv5(self.up5(x4))
        # print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        # >> torch.Size([5, 32, 128]) torch.Size([5, 32, 272]) torch.Size([5, 32, 576]) torch.Size([5, 32, 1216]) torch.Size([5, 32, 2560])

        x_rec = self.conv6(x5)

        return x_rec

class Classifier_cddg_fan(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(in_features=640, out_features=300)
        self.linear2 = nn.Linear(in_features=300, out_features=2)
    def forward(self, x):
        '''
        'x' is the feature vector
        '''
        x1 = self.linear1(x)
        x2 = self.linear2(x1)

        return x2  # logits
