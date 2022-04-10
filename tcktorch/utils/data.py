import numpy as np

class Dataset:

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

class Dataloader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.reset()


    def reset(self):
        n = self.dataset.__len__()
        index_list = list(range(n))
        if self.shuffle:
            np.random.shuffle(index_list)
        self.iter_index_list = []
        if self.drop_last:
            n = n // self.batch_size * self.batch_size
            index_list = index_list[:n]
        for i in range(0,n, self.batch_size):
            self.iter_index_list.append(index_list[i:(i+self.batch_size)])
        self.iter_index_list = list(reversed(self.iter_index_list))

    def __iter__(self):
        return self

    def __next__(self):
        if not self.iter_index_list:
            self.reset()
            raise StopIteration
        index = self.iter_index_list.pop()
        return self.dataset.__getitem__(index)


