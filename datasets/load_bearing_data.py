import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np

import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, freqz, hilbert

from utils.DatasetClass import InfiniteDataLoader, SimpleDataset

###################################
class ReadDataset():
    '''
    test code:
    x =  torch.randn((120000,)).numpy()
    rotating_speed = np.ones(120000)*(1797/60)
    sampling_frequency = 12000
    unified_Os = 350
    the_read_dataset = ReadDataset()
    x_new = the_read_dataset.angular_resample(x, rotating_speed, sampling_frequency, unified_Os)
    '''
    def __init__(self):
        pass


    def normalization_processing(self, data):
        data_mean = data.mean()
        data_std = data.std()

        data = data - data_mean
        data = data / data_std

        return data

    def resize_signal(self, x):
        """
        downsampling or upsampling
        The shape of input x is (120000, 1)
        """
        x = torch.from_numpy(x)
        mode = self.interpolate_mode
        m,n = x.shape
        y = torch.nn.functional.interpolate(torch.reshape(x, (1,n,m)), scale_factor=self.scale_factor, mode = mode)

        return y.view(y.size(2), y.size(1)).numpy()# reshape back

    def butter_lowpass(self, cutoff, fs, order=5):
        return butter(order, cutoff, fs=fs, btype='low', analog=False)

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        '''
        https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
        fs      : sampling rate, Hz
        cutoff  : desired cutoff frequency of the filter, Hz
        order should not be too large. If order=10, the signal maybe useless and the model can not converge
        '''
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)

        return y

    def angular_resample(self, x, rotating_speed, sampling_frequency, unified_Os):
        '''
        Angular resampling:角域重采样
        'rotating_speed' is a numpy vector with the same length of x,
        the unit of 'rotating_speed' should be Hz
        max(angular_position) is the total rad number (总转数)
        max(angular_position)*unified_Os is the number of total sampling points (总采样数)

        test code:
        x =  torch.randn((120000,)).numpy()
        rotating_speed = np.ones(120000)*(1797/60)
        sampling_frequency = 12000
        unified_Os = 350
        '''
        x = np.squeeze(x)
        rotating_speed = np.squeeze(rotating_speed)
        cut_off_frequency = unified_Os*max(rotating_speed)/2
        if cut_off_frequency <= self.sampling_frequency/2.0:
            x = self.butter_lowpass_filter(x, cut_off_frequency, sampling_frequency)
        angular_position = np.cumsum(rotating_speed/sampling_frequency)
        f = interp1d(angular_position, x)
        resample_position = np.linspace(min(angular_position), max(angular_position), np.round(max(angular_position)*unified_Os).astype(np.int64))
        x_new = f(resample_position)

        return x_new


    def load_dataloaders(self):
        the_data = self.read_data_file()

        dataset_train = SimpleDataset(the_data['train'])
        dataset_test  = SimpleDataset(the_data['test'] )

        # dataloader_params_train = dict(batch_size=self.batch_size,
        #                 shuffle=True,
        #                 num_workers=0,  # use main thread only or may receive multiple batches
        #                 drop_last=True,
        #                 pin_memory=False)
        dataloader_params_test = dict(batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=0,  # use main thread only or may receive multiple batches
                        drop_last=False,
                        pin_memory=False)

        train_loader = InfiniteDataLoader(dataset=dataset_train, batch_size=self.batch_size)
        test_loader  = DataLoader(dataset_test,**dataloader_params_test)
        return train_loader, test_loader

class ReadCWRU(ReadDataset):
    """
    2023-01-09:
    - This class is for the domain generalization.
    - The domain label is acutually not needed in the StableNet setting (single domain)
    - But the working condition label is needed in some DG methods. Therefore, the label_wc is added.
    - According to the literature, we need to split each domain into two parts : training part + validation part.
    - During the training stage, we use the training parts of all the source domains to train the model. In each epoch, we test the model performance on the validation set. Theoretially, the target domain dataset can not be used for the model selection.
    - During the inference stage, the target domain is applied to verify the model performance.
    - The training dataset is wrapped by InfiniteDataLoader, the test dataset is wrapped by the default Dataloader

    test code:
    read_cwru = ReadCWRU(configs)
    cwru_train_loader, cwru_test_loader = read_cwru.load_dataloaders()
    cwru_train_iter = iter(cwru_train_loader)
    the_cddg = CDDG(configs)

    currtime = str(time.time())[:10]
    logger = create_logger('IDEA_test2//log_file'+currtime)

    for i in range(5):
        minibatches = next(cwru_train_iter)
        a = the_cddg.update_unified(minibatches)
        logger.info(a)
    """
    def __init__(self, configs):
        self.configs = configs
        self.use_fft = configs.use_fft
        self.use_hilbert_envelope = configs.use_hilbert_envelope
        self.batch_size = configs.batch_size
        self.data_length = configs.data_length
        self.sampling_frequency = 12000 # the sampling frequency of CWRU dataset is 12000 Hz
        self.unified_Os = configs.unified_Os
        # self.unified_fs = configs.unified_fs
        # self.scale_factor = self.unified_fs / self.sampling_frequency
        # self.interpolate_mode = configs.interpolate_mode
        # self.use_resize_signal = configs.use_resize_signal
        self.use_angular_resample = configs.use_angular_resample

        self.num_classes = configs.num_classes
        self.dataset_debug = configs.dataset_debug

    def read_data_file(self):
        if self.dataset_debug:
            datafile_path = '../../Data/CWRU/all_data_DriveEnd'
        else:
            datafile_path = '../Data/CWRU/all_data_DriveEnd'
        data = sio.loadmat(datafile_path)
        rotating_speed_list = [1797, 1772, 1750, 1730] # rad/min

        if self.num_classes == 4:
            category_list = ['NM','IR_007','IR_014','IR_021','OR_007','OR_014','OR_021', 'BA_007','BA_014','BA_021']
            class_idx = [0,1,1,1,2,2,2,3,3,3]  #4 classes
        elif self.num_classes == 3:
            category_list = ['NM','IR_007','IR_014','IR_021','OR_007','OR_014','OR_021']
            class_idx = [0,1,1,1,2,2,2]  #3 classes
        else:
            raise ValueError('The number of classes should be 3 or 4!')

        sample_length = self.data_length # 2048
        signal_length = 120000 # the original length of each signal in CWRU dataset is around 120000
        # if self.use_resize_signal:
        #     signal_length_resized =  np.int64(np.round(signal_length * self.scale_factor)) # the length of each signal is resized to "signal_length * self.scale_factor"
        #     cut_point = np.int64(signal_length_resized/2)-1 # we only use the first half for training and the rest half for validation
        # else:
        #     cut_point = 60000

        sample_num_each_signal = 30
        # start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))

        X_train = []
        y_train = []
        # wc_train = []
        X_test = []
        y_test = []
        # wc_test = []


        cwru_data = {}
        cwru_data_train = {}
        cwru_data_test  = {}

        for i in range(len(class_idx)):
            this_ctgr = category_list[i]
            for j in range(4): # four working conditions
                key_name = this_ctgr + '_' + str(j)
                # constant speed,2023/03/17
                rotating_speed = rotating_speed_list[j]/60 # Hz
                rotating_speed = np.ones(signal_length)*rotating_speed

                this_ctgr_data = data[key_name] #shape (121991, 1)
                this_ctgr_data = this_ctgr_data[:signal_length,:] # shape (120000, 1)
                ########### If the unified_fs is inequal to sampling frequency of the dataset, then we need to resize the signal before segmentation
                # if self.scale_factor !=1 and self.use_resize_signal:
                #     this_ctgr_data = self.resize_signal(this_ctgr_data)
                ###########
                # 2023/03/17
                if self.use_angular_resample:
                    this_ctgr_data = self.angular_resample(this_ctgr_data, rotating_speed, self.sampling_frequency, self.unified_Os)
                    this_ctgr_data = np.expand_dims(this_ctgr_data,1)

                    signal_length_resized = this_ctgr_data.shape[0]
                    cut_point = np.int64(signal_length_resized/2)
                    # print(this_ctgr_data.shape)
                    start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))
                else:
                    cut_point = np.int64(signal_length/2)
                    # print(this_ctgr_data.shape)
                    start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))


                [X_train.append(this_ctgr_data[k:k+sample_length]) for k in start_idx]
                [y_train.append(class_idx[i]) for k in start_idx]
                # [wc_train.append(j) for k in start_idx]

                [X_test.append(this_ctgr_data[k+cut_point:k+cut_point+sample_length]) for k in start_idx]
                [y_test.append(class_idx[i]) for k in start_idx]
                # [wc_test.append(j) for k in start_idx]

        y_train = np.array(y_train)
        y_train = torch.from_numpy(y_train)
        # wc_train = np.array(wc_train)
        # wc_train = torch.from_numpy(wc_train)
        X_train = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_train])
        # print(X_train.shape)

        y_test = np.array(y_test)
        y_test= torch.from_numpy(y_test)
        # wc_test = np.array(wc_test)
        # wc_test = torch.from_numpy(wc_test)
        X_test = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_test])
        # print(X_test.shape)


        if self.configs.use_fft:
            X_train = torch.abs(torch.fft.fft(torch.from_numpy(X_train.astype(np.float32))))#.view(-1,1,sample_length)#.view(-1,1,32,32)
            X_train = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_train.data.numpy()])#[:,:1024]
            # X_train = torch.from_numpy(X_train).view(-1,1,32,32)
            X_train = torch.from_numpy(X_train).view(-1,1,sample_length)
            # debug
            # print(X_train.shape)
            # print(X_test.shape)

            X_test = torch.abs(torch.fft.fft(torch.from_numpy(X_test.astype(np.float32))))#.view(-1,1,sample_length)#.view(-1,1,32,32)

            X_test = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_test.data.numpy()])#[:,:1024]
            # X_test = torch.from_numpy(X_test).view(-1,1,32,32)
            X_test = torch.from_numpy(X_test.astype(np.float32)).view(-1,1,sample_length)

        elif self.use_hilbert_envelope:
            X_train = np.abs(hilbert(X_train)).astype(np.float32)
            X_train = torch.from_numpy(X_train).view(-1,1,sample_length)

            X_test = np.abs(hilbert(X_test)).astype(np.float32)
            X_test = torch.from_numpy(X_test).view(-1,1,sample_length)


        else:
            X_train = torch.from_numpy(X_train.astype(np.float32)).view(-1,1,sample_length)
            X_test  = torch.from_numpy(X_test.astype(np.float32)).view(-1,1,sample_length)

        cwru_data_train['data']  = X_train
        cwru_data_train['label'] = y_train
        # cwru_data_train['label_wc'] = wc_train

        cwru_data_test['data']  = X_test
        cwru_data_test['label'] = y_test
        # cwru_data_test['label_wc'] = wc_test

        cwru_data['train'] = cwru_data_train
        cwru_data['test'] = cwru_data_test

        return cwru_data

class ReadJNU(ReadDataset):
    '''
    The sampling frequency of this dataset is 50kHz
    More details about this dataset can be found in the paper "Sequential Fuzzy Diagnosis Method for Motor Roller Bearing in
    Variable Operating Conditions Based on Vibration Analysis"
    test code:
    read_jnu = ReadJNU(configs)
    jnu_train_loader, jnu_test_loader = read_jnu.load_dataloaders()
    jnu_train_iter = next(iter(jnu_train_loader))
    plt.plot(jnu_train_iter[0][2,0,:].data.numpy())
    '''
    def __init__(self, configs):
        self.configs = configs
        self.use_fft = configs.use_fft
        self.use_hilbert_envelope = configs.use_hilbert_envelope
        self.batch_size = configs.batch_size
        self.data_length = configs.data_length # 2048
        self.sampling_frequency = 50000 # the sampling frequency of MFS dataset is 50000 Hz
        # self.unified_fs = configs.unified_fs
        # self.scale_factor = self.unified_fs / self.sampling_frequency
        # self.interpolate_mode = configs.interpolate_mode
        # self.use_resize_signal = configs.use_resize_signal
        self.use_angular_resample = configs.use_angular_resample
        self.unified_Os = configs.unified_Os
        self.num_classes = configs.num_classes
        self.dataset_debug = configs.dataset_debug

    def read_data_file(self):
        if self.dataset_debug:
            datafile_path = '../../Data/JNU_bearing/JNU_bearing.mat'
        else:
            datafile_path = '../Data/JNU_bearing/JNU_bearing.mat'
        data = sio.loadmat(datafile_path)

        if self.num_classes == 3:
            category_list = ['NM_600','NM_800','NM_1000', 'IR_600','IR_800','IR_1000','OR_600','OR_800','OR_1000'] #3 classes
            class_idx = [0,0,0,1,1,1,2,2,2]  #3 classes
            rotating_speed_list = [600., 800., 1000., 600., 800., 1000.,  600.,  800., 1000.] # rad/min

        elif self.num_classes == 4:
            category_list = ['NM_600', 'NM_800', 'NM_1000', 'IR_600','IR_800','IR_1000','OR_600','OR_800','OR_1000','BA_600','BA_800','BA_1000'] # 4 classes
            class_idx = [0,0,0,1,1,1,2,2,2, 3,3,3]  #4 classes
            rotating_speed_list = [600., 800., 1000., 600., 800., 1000.,  600.,  800., 1000., 600.,  800., 1000.] # rad/min
        else:
            raise ValueError('The number of classes should be 3 or 4!')
        sample_length = self.data_length # 2560, sample length for the model input
        signal_length = 500000  # the original length of each signal in JNU dataset is 500000

        # cut_point = np.int64(signal_length_resized*2/3) # we only use the first half for training and the rest half for validation

        # if self.use_resize_signal:
        #     signal_length_resized =  np.int64(np.round(signal_length * self.scale_factor)) # the length of each signal is resized to "signal_length * self.scale_factor"
        #     cut_point = np.int64(signal_length_resized/2)-1 # we only use the first half for training and the rest half for validation
        # else:
        #     cut_point = 250000

        sample_num_each_signal = 30 # 250000/2560=97.65
        # start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))


        X_train = []
        y_train = []
        # wc_train = []
        X_test = []
        y_test = []
        # wc_test = []

        jnu_data = {}
        jnu_data_train = {}
        jnu_data_test  = {}

        for i in range(len(class_idx)):
            this_ctgr = category_list[i]
            # for j in range(4): # four working conditions
                # key_name = this_ctgr + '_' + str(j)
            this_ctgr_data = data[this_ctgr]
            this_ctgr_data = this_ctgr_data[:signal_length,:]
            ########### If the unified_fs is inequal to sampling frequency of the dataset, then we need to resize the signal before segmentation
            # if self.scale_factor !=1 and self.use_resize_signal:
            #     this_ctgr_data = self.resize_signal(this_ctgr_data)
            # ###########
            # 2023/03/17
            rotating_speed = np.ones(signal_length)*rotating_speed_list[i]/60.0 #Hz

            if self.use_angular_resample:
                this_ctgr_data = self.angular_resample(this_ctgr_data, rotating_speed, self.sampling_frequency, self.unified_Os)
                this_ctgr_data = np.expand_dims(this_ctgr_data,1)

                signal_length_resized = this_ctgr_data.shape[0]
                cut_point = np.int64(signal_length_resized/2)
                # print(this_ctgr_data.shape)
                start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))
            else:
                cut_point = 250000
                start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))

            [X_train.append(this_ctgr_data[k:k+sample_length]) for k in start_idx]
            [y_train.append(class_idx[i]) for k in start_idx]
            # [wc_train.append(j) for k in start_idx]

            [X_test.append(this_ctgr_data[k+cut_point:k+cut_point+sample_length]) for k in start_idx]
            [y_test.append(class_idx[i]) for k in start_idx]
            # [wc_test.append(j) for k in start_idx]

        y_train = np.array(y_train)
        y_train = torch.from_numpy(y_train)
        # wc_train = np.array(wc_train)
        # wc_train = torch.from_numpy(wc_train)
        X_train = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_train])

        y_test = np.array(y_test)
        y_test= torch.from_numpy(y_test)
        # wc_test = np.array(wc_test)
        # wc_test = torch.from_numpy(wc_test)
        X_test = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_test])


        if self.configs.use_fft:
            X_train = torch.abs(torch.fft.fft(torch.from_numpy(X_train.astype(np.float32))))#.view(-1,1,sample_length)#.view(-1,1,32,32)
            X_train = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_train.data.numpy()])#[:,:1024]
            X_train = torch.from_numpy(X_train).view(-1,1,sample_length)
            # X_train = torch.from_numpy(X_train).view(-1,1,32,32)

            X_test = torch.abs(torch.fft.fft(torch.from_numpy(X_test.astype(np.float32))))#.view(-1,1,sample_length)#.view(-1,1,32,32)
            X_test = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_test.data.numpy()])#[:,:1024]
            X_test = torch.from_numpy(X_test).view(-1,1,sample_length)
            # X_test = torch.from_numpy(X_test).view(-1,1,32,32)

        elif self.use_hilbert_envelope:
            X_train = np.abs(hilbert(X_train)).astype(np.float32)
            X_train = torch.from_numpy(X_train).view(-1,1,sample_length)

            X_test = np.abs(hilbert(X_test)).astype(np.float32)
            X_test = torch.from_numpy(X_test).view(-1,1,sample_length)


        else:
            X_train = torch.from_numpy(X_train.astype(np.float32)).view(-1,1,sample_length)
            X_test  = torch.from_numpy(X_test.astype(np.float32)).view(-1,1,sample_length)

        jnu_data_train['data']  = X_train
        jnu_data_train['label'] = y_train
        # cwru_data_train['label_wc'] = wc_train

        jnu_data_test['data']  = X_test
        jnu_data_test['label'] = y_test
        # cwru_data_test['label_wc'] = wc_test

        jnu_data['train'] = jnu_data_train
        jnu_data['test'] = jnu_data_test

        return jnu_data

class ReadUOTTAWA(ReadDataset):
    '''
    The sampling frequency is 200kHz, sampling 10 seconds
    This is a bearing dataset with varying conditions
    There are 3 health conditions: Normal, Inner Race, and Outer Race
    The "UOTTAWA_RS_bearing.mat" contains the information of rotating speed

    '''
    def __init__(self, configs):
        self.configs = configs
        self.use_fft = configs.use_fft
        self.use_hilbert_envelope = configs.use_hilbert_envelope
        self.batch_size = configs.batch_size
        self.data_length = configs.data_length # 2048
        self.sampling_frequency = 200000 # the sampling frequency of MFS dataset is 50000 Hz
        # self.unified_fs = configs.unified_fs
        # self.scale_factor = self.unified_fs / self.sampling_frequency
        # self.interpolate_mode = configs.interpolate_mode
        # self.use_resize_signal = configs.use_resize_signal
        self.use_angular_resample = configs.use_angular_resample
        self.unified_Os = configs.unified_Os
        self.num_classes = configs.num_classes
        self.dataset_debug = configs.dataset_debug

    def read_data_file(self):
        if  self.dataset_debug:
            datafile_path = '../../Data/UOTTAWA_bearing/UOTTAWA_bearing'
            datafile_path_RS = '../../Data/UOTTAWA_bearing/UOTTAWA_RS_bearing'
        else:
            datafile_path = '../Data/UOTTAWA_bearing/UOTTAWA_bearing'
            datafile_path_RS = '../Data/UOTTAWA_bearing/UOTTAWA_RS_bearing'
        data = sio.loadmat(datafile_path)
        data_RS = sio.loadmat(datafile_path_RS)

        if self.num_classes == 4:
            category_list = ['HA1','HB1','HC1','HD1','IA1','IB1','IC1','ID1','OA1','OB1','OC1','OD1']
            class_idx = [0,0,0,0, 1,1,1,1, 2,2,2,2]  #3 classes
            print('The UOTTAWA bearing dataset only has 3 classes')
        elif self.num_classes == 3:
            category_list = ['HA1','HB1','HC1','HD1','IA1','IB1','IC1','ID1','OA1','OB1','OC1','OD1']
            class_idx = [0,0,0,0, 1,1,1,1, 2,2,2,2]  #3 classes
        else:
            raise ValueError('The number of classes should be 3 or 4!')

        sample_length = self.data_length # 2048
        signal_length = 2000000 # the original length of each signal in UOTTAWA dataset is 2000000 (10 seconds)
        # if self.use_resize_signal:
        #     signal_length_resized =  np.int64(np.round(signal_length * self.scale_factor)) # the length of each signal is resized to "signal_length * self.scale_factor"
        #     cut_point = np.int64(signal_length_resized/2)-1 # we only use the first half for training and the rest half for validation
        # else:
        #     cut_point = 1000000

        sample_num_each_signal = 30
        # start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))


        X_train = []
        y_train = []
        # wc_train = []
        X_test = []
        y_test = []
        # wc_test = []


        uot_data = {}
        uot_data_train = {}
        uot_data_test  = {}

        for i in range(len(class_idx)):
            this_ctgr = category_list[i]
            # for j in range(4): # four working conditions
                # key_name = this_ctgr + '_' + str(j)
            this_ctgr_data = data[this_ctgr] #shape (121991, 1)
            this_ctgr_data_RS = data_RS[this_ctgr+'_RS']
            this_ctgr_data = this_ctgr_data[:signal_length,:] # shape (120000, 1)
            ########### If the unified_fs is inequal to sampling frequency of the dataset, then we need to resize the signal before segmentation
            # if self.scale_factor !=1 and self.use_resize_signal:
            #     this_ctgr_data = self.resize_signal(this_ctgr_data)
            # ###########
            rotating_speed = this_ctgr_data_RS #np.ones(signal_length)*rotating_speed_list[i]/60.0 #Hz

            if self.use_angular_resample:
                this_ctgr_data = self.angular_resample(this_ctgr_data, rotating_speed, self.sampling_frequency, self.unified_Os)
                this_ctgr_data = np.expand_dims(this_ctgr_data,1)

                signal_length_resized = this_ctgr_data.shape[0]
                cut_point = np.int64(signal_length_resized/2)
                # print(this_ctgr_data.shape)
                start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))
            else:
                cut_point = 1000000
                start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))


            [X_train.append(this_ctgr_data[k:k+sample_length]) for k in start_idx]
            [y_train.append(class_idx[i]) for k in start_idx]
            # [wc_train.append(j) for k in start_idx]

            [X_test.append(this_ctgr_data[k+cut_point:k+cut_point+sample_length]) for k in start_idx]
            [y_test.append(class_idx[i]) for k in start_idx]
            # [wc_test.append(j) for k in start_idx]

        y_train = np.array(y_train)
        y_train = torch.from_numpy(y_train)
        # wc_train = np.array(wc_train)
        # wc_train = torch.from_numpy(wc_train)
        X_train = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_train])

        y_test = np.array(y_test)
        y_test= torch.from_numpy(y_test)
        # wc_test = np.array(wc_test)
        # wc_test = torch.from_numpy(wc_test)
        X_test = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_test])


        if self.configs.use_fft:
            X_train = torch.abs(torch.fft.fft(torch.from_numpy(X_train.astype(np.float32))))#.view(-1,1,sample_length)#.view(-1,1,32,32)
            X_train = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_train.data.numpy()])#[:,:1024]
            # X_train = torch.from_numpy(X_train).view(-1,1,32,32)
            X_train = torch.from_numpy(X_train).view(-1,1,sample_length)

            X_test = torch.abs(torch.fft.fft(torch.from_numpy(X_test.astype(np.float32))))#.view(-1,1,sample_length)#.view(-1,1,32,32)
            X_test = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_test.data.numpy()])#[:,:1024]
            X_test = torch.from_numpy(X_test)#.view(-1,1,32,32)
            X_test = torch.from_numpy(X_test).view(-1,1,sample_length)

        elif self.use_hilbert_envelope:
            X_train = np.abs(hilbert(X_train)).astype(np.float32)
            X_train = torch.from_numpy(X_train).view(-1,1,sample_length)

            X_test = np.abs(hilbert(X_test)).astype(np.float32)
            X_test = torch.from_numpy(X_test).view(-1,1,sample_length)


        else:
            X_train = torch.from_numpy(X_train.astype(np.float32)).view(-1,1,sample_length)
            X_test  = torch.from_numpy(X_test.astype(np.float32)).view(-1,1,sample_length)

        uot_data_train['data']  = X_train
        uot_data_train['label'] = y_train
        # cwru_data_train['label_wc'] = wc_train

        uot_data_test['data']  = X_test
        uot_data_test['label'] = y_test
        # cwru_data_test['label_wc'] = wc_test

        uot_data['train'] = uot_data_train
        uot_data['test'] = uot_data_test

        return uot_data

class ReadMFPT(ReadDataset):
    '''
    The rotating frequency of the shaft is 25Hz
    The sampling frequency is different: normal condition: 97656Hz, inner/outer race fault: 48828Hz, I resample the normal condition into 48828Hz
    This is a bearing dataset with varying loads:
        normal: 270lbs
        outer race: 25, 50, 100, 150, 200, 250, 300lbs
        inner race: 0,  50, 100, 150, 200, 250, 300lbs
    We only use the following six conditions [50, 100, 150, 200, 250, 300]
    Each sample has a time duration of 3 seconds.
    And, all the data are collected under the input shaft rate of 25 Hz

    There are 3 health conditions: Normal, Inner Race, and Outer Race

    test code
    read_mfpt = ReadMFPT(configs)
    mfpt_train_loader, mfpt_test_loader = read_mfpt.load_dataloaders()
    mfpt_train_iter = next(iter(mfpt_train_loader))
    plt.plot(mfpt_train_iter[0][1,0,:].data.numpy())

    '''
    def __init__(self, configs):
        self.configs = configs
        self.use_fft = configs.use_fft
        self.use_hilbert_envelope = configs.use_hilbert_envelope
        self.batch_size = configs.batch_size
        self.data_length = configs.data_length # 2048
        self.sampling_frequency = 48828 # the sampling frequency of MFPT dataset is 48828 Hz
        # self.unified_fs = configs.unified_fs
        # self.scale_factor = self.unified_fs / self.sampling_frequency
        # self.interpolate_mode = configs.interpolate_mode
        # self.use_resize_signal = configs.use_resize_signal
        self.use_angular_resample = configs.use_angular_resample
        self.unified_Os = configs.unified_Os
        self.num_classes = configs.num_classes
        self.dataset_debug = configs.dataset_debug

    def read_data_file(self):
        if self.dataset_debug:
            datafile_path = '../../Data/MFPT_bearing/MFPT_bearing.mat'
        else:
            datafile_path = '../Data/MFPT_bearing/MFPT_bearing.mat'
        data = sio.loadmat(datafile_path)
        rotating_speed_all = 25 # Hz

        if self.num_classes == 3:
            category_list = ['NM_1','NM_2','NM_3', 'IR_50','IR_100','IR_150','IR_200','IR_250','IR_300', 'OR_50','OR_100','OR_150','OR_200','OR_250','OR_300'] #3 classes
            class_idx = [0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2]  #3 classes
        elif self.num_classes == 4:
            category_list = ['NM_1','NM_2','NM_3', 'IR_50','IR_100','IR_150','IR_200','IR_250','IR_300', 'OR_50','OR_100','OR_150','OR_200','OR_250','OR_300'] #3 classes
            class_idx = [0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2]  #3 classes
            print('The MFPT bearing dataset only has 3 classes')
        else:
            raise ValueError('The number of classes should be 3 or 4!')
        sample_length = self.data_length # 2560, sample length for the model input
        signal_length = 140000  # the original length of each signal in MFPT dataset is 146484

        # cut_point = np.int64(signal_length_resized*2/3) # we only use the first half for training and the rest half for validation

        # if self.use_resize_signal:
        #     signal_length_resized =  np.int64(np.round(signal_length * self.scale_factor)) # the length of each signal is resized to "signal_length * self.scale_factor"
        #     cut_point = np.int64(signal_length_resized/2)-1 # we only use the first half for training and the rest half for validation
        # else:
        #     cut_point = 70000

        sample_num_each_signal = 10 # 3 second is enough for 10 samples
        # start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))


        X_train = []
        y_train = []
        # wc_train = []
        X_test = []
        y_test = []
        # wc_test = []


        mfpt_data = {}
        mfpt_data_train = {}
        mfpt_data_test  = {}

        for i in range(len(class_idx)):
            this_ctgr = category_list[i]
            # for j in range(4): # four working conditions
                # key_name = this_ctgr + '_' + str(j)
            this_ctgr_data = data[this_ctgr]
            this_ctgr_data = this_ctgr_data[:signal_length,:]
            rotating_speed = np.ones(signal_length)*rotating_speed_all
            # ########### If the unified_fs is inequal to sampling frequency of the dataset, then we need to resize the signal before segmentation
            # if self.scale_factor !=1 and self.use_resize_signal:
            #     this_ctgr_data = self.resize_signal(this_ctgr_data)
            # ###########
            if self.use_angular_resample:
                this_ctgr_data = self.angular_resample(this_ctgr_data, rotating_speed, self.sampling_frequency, self.unified_Os)
                this_ctgr_data = np.expand_dims(this_ctgr_data,1)

                signal_length_resized = this_ctgr_data.shape[0]
                cut_point = np.int64(signal_length_resized/2)
                # print(this_ctgr_data.shape)
                start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))
            else:
                cut_point = 1000000
                start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))

            [X_train.append(this_ctgr_data[k:k+sample_length]) for k in start_idx]
            [y_train.append(class_idx[i]) for k in start_idx]
            # [wc_train.append(j) for k in start_idx]

            [X_test.append(this_ctgr_data[k+cut_point:k+cut_point+sample_length]) for k in start_idx]
            [y_test.append(class_idx[i]) for k in start_idx]
            # [wc_test.append(j) for k in start_idx]

        y_train = np.array(y_train)
        y_train = torch.from_numpy(y_train)
        # wc_train = np.array(wc_train)
        # wc_train = torch.from_numpy(wc_train)
        X_train = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_train])

        y_test = np.array(y_test)
        y_test= torch.from_numpy(y_test)
        # wc_test = np.array(wc_test)
        # wc_test = torch.from_numpy(wc_test)
        X_test = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_test])


        if self.configs.use_fft:
            X_train = torch.abs(torch.fft.fft(torch.from_numpy(X_train.astype(np.float32))))#.view(-1,1,sample_length)#.view(-1,1,32,32)
            X_train = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_train.data.numpy()])[:,:1024]
            X_train = torch.from_numpy(X_train).view(-1,1,32,32)
            # X_train = torch.from_numpy(X_train).view(-1,1,sample_length)

            X_test = torch.abs(torch.fft.fft(torch.from_numpy(X_test.astype(np.float32))))#.view(-1,1,sample_length)#.view(-1,1,32,32)
            X_test = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_test.data.numpy()])[:,:1024]
            X_test = torch.from_numpy(X_test).view(-1,1,32,32)
            # X_test = torch.from_numpy(X_test).view(-1,1,sample_length)

        elif self.use_hilbert_envelope:
            X_train = np.abs(hilbert(X_train)).astype(np.float32)
            X_train = torch.from_numpy(X_train).view(-1,1,sample_length)

            X_test = np.abs(hilbert(X_test)).astype(np.float32)
            X_test = torch.from_numpy(X_test).view(-1,1,sample_length)

        else:
            X_train = torch.from_numpy(X_train.astype(np.float32)).view(-1,1,sample_length)
            X_test  = torch.from_numpy(X_test.astype(np.float32)).view(-1,1,sample_length)

        mfpt_data_train['data']  = X_train
        mfpt_data_train['label'] = y_train
        # cwru_data_train['label_wc'] = wc_train

        mfpt_data_test['data']  = X_test
        mfpt_data_test['label'] = y_test
        # cwru_data_test['label_wc'] = wc_test

        mfpt_data['train'] = mfpt_data_train
        mfpt_data['test']  = mfpt_data_test

        return mfpt_data

class ReadPU(ReadDataset):
    '''
    Paderborn University bearing dataset
    The sampling frequency is 64kHz
    This is a very big dataset, here I just select some folders

    '''

    def __init__(self, configs):
        self.configs = configs
        self.use_fft = configs.use_fft
        self.use_hilbert_envelope = configs.use_hilbert_envelope
        self.batch_size = configs.batch_size
        self.data_length = configs.data_length # 2048
        self.sampling_frequency = 64000 # the sampling frequency of PU dataset is 64kHz
        # self.unified_fs = configs.unified_fs
        # self.scale_factor = self.unified_fs / self.sampling_frequency
        # self.interpolate_mode = configs.interpolate_mode
        # self.use_resize_signal = configs.use_resize_signal
        self.use_angular_resample = configs.use_angular_resample
        self.unified_Os = configs.unified_Os
        self.num_classes = configs.num_classes
        self.dataset_debug = configs.dataset_debug

    def read_data_file(self):
        if self.dataset_debug:
            datafile_path = '../../Data/Paderborn/PaderbornBearing'
        else:
            datafile_path = '../Data/Paderborn/PaderbornBearing'
        data = sio.loadmat(datafile_path)
        category_list = ['health','inner','outer']
        working_condition_list = ['N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10']
        rotating_speed_list = [900, 1500, 1500, 1500]
        file_list = {'health':['K001','K002','K003'],  'inner':['KI04', 'KI14','KI16'],'outer':['KA04','KA15','KA22']}

        # if self.num_classes == 3:
        #     category_list = ['NM_1','NM_2','NM_3', 'IR_50','IR_100','IR_150','IR_200','IR_250','IR_300', 'OR_50','OR_100','OR_150','OR_200','OR_250','OR_300'] #3 classes
        #     class_idx = [0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2]  #3 classes
        # elif self.num_classes == 4:
        #     category_list = ['NM_1','NM_2','NM_3', 'IR_50','IR_100','IR_150','IR_200','IR_250','IR_300', 'OR_50','OR_100','OR_150','OR_200','OR_250','OR_300'] #3 classes
        #     class_idx = [0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2]  #3 classes
        #     print('The Paderborn bearing dataset only has 3 classes')
        # else:
        #     raise ValueError('The number of classes should be 3 or 4!')
        sample_length = self.data_length # 2560, sample length for the model input
        signal_length = 250000  # the original length of each signal in PU dataset is around 250000

        # cut_point = np.int64(signal_length_resized*2/3) # we only use the first half for training and the rest half for validation

        # if self.use_resize_signal:
        #     signal_length_resized =  np.int64(np.round(signal_length * self.scale_factor)) # the length of each signal is resized to "signal_length * self.scale_factor"
        #     cut_point = np.int64(signal_length_resized/2)-1 # we only use the first half for training and the rest half for validation
        # else:
        #     cut_point = 120000

        sample_num_each_signal = 10 # 3 second is enough for 10 samples
        # start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))


        X_train = []
        y_train = []
        # wc_train = []
        X_test = []
        y_test = []
        # wc_test = []


        pu_data = {}
        pu_data_train = {}
        pu_data_test  = {}

        for i in range(len(category_list)):
            this_ctgr = category_list[i]

            for j in range(len(working_condition_list)):
                this_working_condition = working_condition_list[j]
                rotating_speed = np.ones(signal_length)*rotating_speed_list[j]/60.0 #Hz

                for jj in range(3):
                    file_names = file_list[this_ctgr]
                    file_name = file_names[jj]
                    key_name  = this_working_condition + '_' + file_name
                    this_ctgr_data = data[key_name]
                    # this_ctgr_data = np.reshape(this_ctgr_data, (this_ctgr_data.shape[1],this_ctgr_data.shape[0]))
                    this_ctgr_data =  this_ctgr_data[:,:signal_length]

                    if self.use_angular_resample:
                        # print(this_ctgr_data.shape)
                        # print(rotating_speed.shape)
                        this_ctgr_data = self.angular_resample(this_ctgr_data, rotating_speed, self.sampling_frequency, self.unified_Os)
                        this_ctgr_data = np.expand_dims(this_ctgr_data,1)

                        signal_length_resized = this_ctgr_data.shape[0]
                        cut_point = np.int64(signal_length_resized/2)
                        # print(this_ctgr_data.shape)
                        start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))
                    else:
                        cut_point = 125000
                        start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))

                    # this_ctgr = category_list[i]
                    # for j in range(4): # four working conditions
                        # key_name = this_ctgr + '_' + str(j)
                    # this_ctgr_data = data[this_ctgr]
                    # this_ctgr_data = this_ctgr_data[:signal_length,:]
                    ########### If the unified_fs is inequal to sampling frequency of the dataset, then we need to resize the signal before segmentation
                    # if self.scale_factor !=1 and self.use_resize_signal:
                    #     this_ctgr_data = self.resize_signal(this_ctgr_data)
                    # ###########

                    [X_train.append(this_ctgr_data[k:k+sample_length]) for k in start_idx]
                    [y_train.append(i) for k in start_idx]
                    # [wc_train.append(j) for k in start_idx]

                    [X_test.append(this_ctgr_data[k+cut_point:k+cut_point+sample_length]) for k in start_idx]
                    [y_test.append(i) for k in start_idx]
                    # [wc_test.append(j) for k in start_idx]

        y_train = np.array(y_train)
        y_train = torch.from_numpy(y_train)
        # wc_train = np.array(wc_train)
        # wc_train = torch.from_numpy(wc_train)
        X_train = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_train])

        y_test = np.array(y_test)
        y_test= torch.from_numpy(y_test)
        # wc_test = np.array(wc_test)
        # wc_test = torch.from_numpy(wc_test)
        X_test = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_test])

        if self.configs.use_fft:
            X_train = torch.abs(torch.fft.fft(torch.from_numpy(X_train.astype(np.float32))))#.view(-1,1,sample_length)#.view(-1,1,32,32)
            X_train = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_train.data.numpy()])#[:,:1024]
            # X_train = torch.from_numpy(X_train).view(-1,1,32,32)
            X_train = torch.from_numpy(X_train).view(-1,1,sample_length)

            X_test = torch.abs(torch.fft.fft(torch.from_numpy(X_test.astype(np.float32))))#.view(-1,1,sample_length)#.view(-1,1,32,32)
            X_test = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_test.data.numpy()])#[:,:1024]
            # X_test = torch.from_numpy(X_test).view(-1,1,32,32)
            X_test = torch.from_numpy(X_test).view(-1,1,sample_length)

        elif self.use_hilbert_envelope:
            X_train = np.abs(hilbert(X_train)).astype(np.float32)
            X_train = torch.from_numpy(X_train).view(-1,1,sample_length)

            X_test = np.abs(hilbert(X_test)).astype(np.float32)
            X_test = torch.from_numpy(X_test).view(-1,1,sample_length)


        else:
            X_train = torch.from_numpy(X_train.astype(np.float32)).view(-1,1,sample_length)
            X_test  = torch.from_numpy(X_test.astype(np.float32)).view(-1,1,sample_length)

        pu_data_train['data']  = X_train
        pu_data_train['label'] = y_train
        # cwru_data_train['label_wc'] = wc_train

        pu_data_test['data']  = X_test
        pu_data_test['label'] = y_test
        # cwru_data_test['label_wc'] = wc_test

        pu_data['train'] = pu_data_train
        pu_data['test']  = pu_data_test

        return pu_data

class ReadDZLRSB(ReadDataset):
    '''
    The sampling frequency of DZLRSB dataset is also 25.6kHz
    The dataset only has 3 classes.
    This is also a highly-qualified bearing dataset (because its clear impulses)
    尽管DZLRSB数据集是具有显著特征的数据集,但是在训练过程中发现明显的先上升后下降过程
    通过使用排除法,发现除了CWRU和MFS数据集以外的其他数据集,都会对其产生负迁移现象
    这种负迁移十分明显,以至于在经过200个step之后就会让模型完全陷入随机猜测,也即30%左右的acc
    '''
    def __init__(self, configs):
        self.configs = configs
        self.use_fft = configs.use_fft
        self.use_hilbert_envelope = configs.use_hilbert_envelope
        self.batch_size = configs.batch_size
        self.data_length = configs.data_length # 2048
        self.sampling_frequency = 25600 # the sampling frequency of DZLRSB dataset is 50000 Hz
        # self.unified_fs = configs.unified_fs
        # self.scale_factor = self.unified_fs / self.sampling_frequency
        # self.interpolate_mode = configs.interpolate_mode
        # self.use_resize_signal = configs.use_resize_signal
        self.use_angular_resample = configs.use_angular_resample
        self.unified_Os = configs.unified_Os
        self.num_classes = configs.num_classes
        self.dataset_debug = configs.dataset_debug

    def read_data_file(self):
        if self.dataset_debug:
            datafile_path = '../../Data/DZLRSB_bearing/DZLRSB_bearing.mat'
        else:
            datafile_path = '../Data/DZLRSB_bearing/DZLRSB_bearing.mat'
        data = sio.loadmat(datafile_path)

        if self.num_classes == 3:
            category_list = [ 'NM_05_060', 'NM_05_500', 'NM_88_060', 'NM_88_500','IR_05_060_10', 'IR_05_060_21', 'IR_05_060_38', 'IR_05_500_10', 'IR_05_500_21', 'IR_05_500_38', 'IR_88_060_10', 'IR_88_060_21', 'IR_88_060_38', 'IR_88_500_10', 'IR_88_500_21', 'IR_88_500_38',  'OR_05_060_14', 'OR_05_060_24', 'OR_05_060_40', 'OR_05_500_14', 'OR_05_500_24', 'OR_05_500_40', 'OR_88_060_14', 'OR_88_060_24', 'OR_88_060_40', 'OR_88_500_14', 'OR_88_500_24', 'OR_88_500_40']#3 classes


            class_idx = [0]*4+[1]*12+[2]*12  #3 classes
        elif self.num_classes == 4:
            category_list = [ 'NM_05_060', 'NM_05_500', 'NM_88_060', 'NM_88_500','IR_05_060_10', 'IR_05_060_21', 'IR_05_060_38', 'IR_05_500_10', 'IR_05_500_21', 'IR_05_500_38', 'IR_88_060_10', 'IR_88_060_21', 'IR_88_060_38', 'IR_88_500_10', 'IR_88_500_21', 'IR_88_500_38',  'OR_05_060_14', 'OR_05_060_24', 'OR_05_060_40', 'OR_05_500_14', 'OR_05_500_24', 'OR_05_500_40', 'OR_88_060_14', 'OR_88_060_24', 'OR_88_060_40', 'OR_88_500_14', 'OR_88_500_24', 'OR_88_500_40']#3 classes
            class_idx = [0]*4+[1]*12+[2]*12  #3 classes
            print('The DZLRSB bearing dataset only has 3 classes (normal, inner race fault, and outer race fault).')
        else:
            raise ValueError('The number of classes should be 3 or 4!')

        rotating_speed_list = []
        for i in category_list:
            # if i[0:2]=='NM':
            if i[6]=='5':
                rotating_speed_list.append(500)
            elif i[7]=='6':
                rotating_speed_list.append(60)

        sample_length = self.data_length # 2560, sample length for the model input
        signal_length = 768000  # the original length of each signal in DZLRSB dataset is set as 512000

        # cut_point = np.int64(signal_length_resized*2/3) # we only use the first half for training and the rest half for validation

        # if self.use_resize_signal:
        #     signal_length_resized =  np.int64(np.round(signal_length * self.scale_factor)) # the length of each signal is resized to "signal_length * self.scale_factor"
        #     cut_point = np.int64(signal_length_resized/2)-1 # we only use the first half for training and the rest half for validation
        # else:
        #     cut_point = 384000

        sample_num_each_signal = 30 # 250000/2560=97.65
        # start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))


        X_train = []
        y_train = []
        # wc_train = []
        X_test = []
        y_test = []
        # wc_test = []


        dzlrsb_data = {}
        dzlrsb_data_train = {}
        dzlrsb_data_test  = {}

        for i in range(len(class_idx)):
            this_ctgr = category_list[i]
            # for j in range(4): # four working conditions
                # key_name = this_ctgr + '_' + str(j)
            this_ctgr_data = data[this_ctgr]
            this_ctgr_data = np.reshape(this_ctgr_data, (this_ctgr_data.shape[1],this_ctgr_data.shape[0]))
            this_ctgr_data = this_ctgr_data[:signal_length,:]
            # ########### If the unified_fs is inequal to sampling frequency of the dataset, then we need to resize the signal before segmentation
            # if self.scale_factor !=1 and self.use_resize_signal:
            #     this_ctgr_data = self.resize_signal(this_ctgr_data)
            # ###########
            rotating_speed = np.ones(signal_length)*rotating_speed_list[i]/60
            if self.use_angular_resample:
                # print(this_ctgr_data.shape)
                # print(rotating_speed.shape)
                this_ctgr_data = self.angular_resample(this_ctgr_data, rotating_speed, self.sampling_frequency, self.unified_Os)
                this_ctgr_data = np.expand_dims(this_ctgr_data,1)

                signal_length_resized = this_ctgr_data.shape[0]
                cut_point = np.int64(signal_length_resized/2)
                # print(this_ctgr_data.shape)
                start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))
            else:
                cut_point = 384000
                start_idx = np.int64(np.round(np.linspace(0,cut_point-sample_length,sample_num_each_signal)))

            [X_train.append(this_ctgr_data[k:k+sample_length]) for k in start_idx]
            [y_train.append(class_idx[i]) for k in start_idx]
            # [wc_train.append(j) for k in start_idx]

            [X_test.append(this_ctgr_data[k+cut_point:k+cut_point+sample_length]) for k in start_idx]
            [y_test.append(class_idx[i]) for k in start_idx]
            # [wc_test.append(j) for k in start_idx]

        y_train = np.array(y_train)
        y_train = torch.from_numpy(y_train)
        # wc_train = np.array(wc_train)
        # wc_train = torch.from_numpy(wc_train)
        X_train = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_train])

        y_test = np.array(y_test)
        y_test= torch.from_numpy(y_test)
        # wc_test = np.array(wc_test)
        # wc_test = torch.from_numpy(wc_test)
        X_test = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_test])


        if self.configs.use_fft:
            X_train = torch.abs(torch.fft.fft(torch.from_numpy(X_train.astype(np.float32))))#.view(-1,1,sample_length)#.view(-1,1,32,32)
            X_train = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_train.data.numpy()])[:,:1024]
            X_train = torch.from_numpy(X_train).view(-1,1,32,32)
            # X_train = torch.from_numpy(X_train).view(-1,1,sample_length)

            X_test = torch.abs(torch.fft.fft(torch.from_numpy(X_test.astype(np.float32))))#.view(-1,1,sample_length)#.view(-1,1,32,32)
            X_test = np.array([self.normalization_processing(np.squeeze(xx)) for xx in X_test.data.numpy()])[:,:1024]
            X_test = torch.from_numpy(X_test).view(-1,1,32,32)
            # X_test = torch.from_numpy(X_test).view(-1,1,sample_length)

        elif self.use_hilbert_envelope:
            X_train = np.abs(hilbert(X_train)).astype(np.float32)
            X_train = torch.from_numpy(X_train).view(-1,1,sample_length)

            X_test = np.abs(hilbert(X_test)).astype(np.float32)
            X_test = torch.from_numpy(X_test).view(-1,1,sample_length)


        else:
            X_train = torch.from_numpy(X_train.astype(np.float32)).view(-1,1,sample_length)
            X_test  = torch.from_numpy(X_test.astype(np.float32)).view(-1,1,sample_length)

        dzlrsb_data_train['data']  = X_train
        dzlrsb_data_train['label'] = y_train
        # cwru_data_train['label_wc'] = wc_train

        dzlrsb_data_test['data']  = X_test
        dzlrsb_data_test['label'] = y_test
        # cwru_data_test['label_wc'] = wc_test

        dzlrsb_data['train'] = dzlrsb_data_train
        dzlrsb_data['test'] = dzlrsb_data_test

        return dzlrsb_data

