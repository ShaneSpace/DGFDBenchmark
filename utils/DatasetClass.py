import torch
from torch.utils.data import DataLoader, Dataset


class _InfiniteSampler(torch.utils.data.Sampler):
    '''
    Wraps another Sampler to yield an infinite stream.
    Copied from DDG code
    '''
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

##
class InfiniteDataLoader:
    '''
    https://blog.csdn.net/loveliuzz/article/details/108756253
    注意这个class的写法, dataloader本身有两种访问方式:
    (1) dataloader 本质上是一个可迭代对象，使用 iter() 访问，不能使用 next() 访问，它本身就是一个可迭代对象， 使用  for i, data in enumerate(dataloader) 来访问。
    (2) 先使用 iter 对 dataloader 进行第一步包装，使用 iter(dataloader) 返回的是一个迭代器，然后可以使用 next 访问。
    显然,这个class使用的是第二种方式,在__init__()中用iter()包装dataloader,然后在__iter__()中使用next访问. 还需要注意yield的随用随取的特性,要不然就是死循环了

    Copied from DDG code
    '''
    def __init__(self, dataset, batch_size=128, weights=None, num_workers=0):
        super().__init__()

        if weights:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=True)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator) #虽然我感觉这个应该是属于__next__()方法的内容

    def __len__(self):
        raise ValueError('This is a infinite dataloader!')


class SimpleDataset(Dataset):
    # 注意CCN中，数据集内部不需要考虑domain label，在unzip时加上domain label即可
    def __init__(self, data_content):
        '''
        The 'data_content' should be a dict with two keys ('data' and 'label')
        ################# test code ##################
        real_data = torch.rand(115,1024)
        real_label = torch.randint(0,4,(115,))
        data_content  = dict(data = real_data, label=real_label)

        my_dataset = SimpleDataset(data_content)
        data_loader_params = dict(batch_size=32,
                                shuffle=True,
                                num_workers=0,  # use main thread only or may receive multiple batches
                                pin_memory=False)

        my_dataloader = DataLoader(my_dataset, **data_loader_params)
        for i_batch, sample_batched in enumerate(my_dataloader):
            print(i_batch, sample_batched)
        ############## End of the test code ##############
        '''
        self.data  = data_content['data']
        self.class_label  = data_content['label']

    def __getitem__(self, index):
        data_i = self.data[index]
        class_label_i = self.class_label[index]

        return data_i, class_label_i

    def __len__(self):
        if self.data.shape[0] != self.class_label.shape[0]:
            raise ValueError('The number of samples should be equal to the number of labels!')
        data_length = self.data.shape[0]
        return data_length