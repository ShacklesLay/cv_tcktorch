import numpy as np
import os
import gzip
from tcktorch.utils.data import Dataset, Dataloader
from tcktorch import nn
from tcktorch import optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from tcktorch.torchvision.Transform import *


def load_data(data_folder):
  files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
  ]

  paths = []
  for fname in files:
    paths.append(os.path.join(data_folder,fname))

  with gzip.open(paths[0], 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[1], 'rb') as imgpath:
    x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

  with gzip.open(paths[2], 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[3], 'rb') as imgpath:
    x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

  return (x_train, y_train), (x_test, y_test)

def crossvalid(dataset):
    x,y = dataset
    train_index = np.random.choice(len(x), round(len(x) * 0.8), replace=False)
    valid_index = np.array(list(set(range(len(x))) - set(train_index)))
    return (x[train_index], y[train_index]), (x[valid_index], y[valid_index])

(train_images, train_labels), (test_images, test_labels) = load_data('minist/')
(train_images, train_labels),(valid_images, valid_labels) = crossvalid((train_images, train_labels))

class TrainData(Dataset):
    def __init__(self, transform=[Resize(), Rotation(), Translation()], p=0):
        data = (train_images,train_labels)
        x = data[0]/255  #归一化可以加速拟合
        self.x = x.reshape(x.shape[0],28,28,1)
        y = data[1]
        self.y = np.eye(10)[y].reshape(y.shape[0], 10)  #one-hot array便于计算损失
        self.transform = transform
        self.p = p


    def __getitem__(self, index):
        x = self.x[index].copy()
        if self.transform:
            p = np.random.rand(x.shape[0])
            select = np.where(p < self.p)[0]
            for i in select:
                t = np.random.choice(self.transform)
                x[i] = t(x[i])
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

class ValidData(Dataset):
    def __init__(self):
        data = (valid_images, valid_labels)
        x = data[0]/255   #归一化可以加速拟合
        self.x = x.reshape(x.shape[0], 28, 28, 1)
        y = data[1]
        self.y = y.reshape(y.shape[0],1)      #测试集不需要计算损失，所以不用变为one-hot array

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

def accuracy(y_hat, y, one_hot = True):
    '''

    :param y_hat: 模型输出
    :param y: 数据集label
    :param one_hot: 标志 y 是否为One-hot
    :return: 正确分类的个数
    '''
    y_hat_trans = np.argmax(y_hat,axis=1).reshape(y_hat.shape[0], 1)
    y_trans = y
    if one_hot:
        y_trans = np.argmax(y,axis=1).reshape(y.shape[0], 1)
    return np.sum(y_hat_trans == y_trans)


def train(model, epoch_num=35, lr=0.1, batch_size=64, seed=37, plot=True, record = False, save = False, weight_decay=0, hidden_units=128):
    np.random.seed(seed)
    optimizer = optim.GD(lr=lr, nnModel=model, weight_decay=weight_decay)
    costfunc = nn.functional.CrossEntropyLoss(nnModel=model)
    traindata = TrainData()
    validdata = ValidData()
    trainloder = Dataloader(dataset=traindata, batch_size=batch_size)
    validloder = Dataloader(dataset=validdata, batch_size=batch_size)
    costs = []
    train_acc = []
    valid_acc = []
    train_time = []
    valid_time = []
    waste = False

    for epoch in range(epoch_num):
        accn = 0
        cost = 0
        time_s = time.time()
        for x_train, ytarget_train in tqdm(trainloder):
            y_train = model(x_train)
            result = costfunc(y_train, ytarget_train)
            costs.append(result.loss)
            cost += result.loss
            accn += accuracy(y_train, ytarget_train)
            result.backward()
            optimizer.step(epoch)
        train_acc.append(accn/len(traindata))
        train_time.append(time.time() - time_s)

        time_s = time.time()
        accn = 0
        for x_valid, ytarget_valid in tqdm(validloder):
            y_valid = model(x_valid, eval_pattern=True)
            accn += accuracy(y_valid, ytarget_valid, one_hot=False)
        valid_acc.append(accn/len(validdata))
        valid_time.append(time.time() - time_s)
        if epoch > 7 and valid_acc[-1]<0.9:
            waste=True
            break
        if epoch > 19 and valid_acc[-1]<0.95:
            waste = True
            break

        print("Epoch:{}\tcost:{}\ttrain_acc:{}\tvalid_acc:{}".format(epoch+1, cost, train_acc[epoch], valid_acc[epoch]))

    if record:
        path = os.path.join('log', 'baseline{}epoch_num{}'.format(lr, epoch_num))
        if not os.path.exists(path):
            os.makedirs(path)
        accs = np.array([train_acc, valid_acc])
        time_cost = np.array([train_time, valid_time])
        np.save(os.path.join(path, 'cost.npy'), costs)
        np.save(os.path.join(path, 'acc.npy'), accs)
        np.save(os.path.join(path, 'time.npy'), time_cost)

    if plot:
        plt.plot(list(range(1, len(costs) + 1)), costs)
        plt.title('Loss Curve')
        plt.ylabel('Loss')
        plt.xlabel('Iter num')
        plt.show()
        plt.plot(list(range(1, len(train_acc) + 1)), 1 - np.array(train_acc), color='blue', label='train')
        plt.plot(list(range(1, len(valid_acc) + 1)), 1 - np.array(valid_acc), color='red', label='valid')
        plt.legend()
        plt.title('Error Curve. Best performance on valid is {}%'.format(max(valid_acc) * 100))
        plt.ylabel('error')
        plt.xlabel('epoch')
        plt.show()

    if save and not waste :
        path = os.path.join('model2', 'lr{}hiddenunits:{}weightdecay:{}'.format(lr, hidden_units, weight_decay))
        if not os.path.exists(path):
            os.makedirs(path)
        path1 = os.path.join(path, 'model.pkl')
        model.save_state_dict(path1)
    return max(valid_acc) * 100


def execute(hidden_units: object = 128, learning_rate: object = 0.1, weight_decay: object = 1e-5) -> object:

    model = nn.Sequential([ nn.functional.Flatten(),
                              nn.functional.Linear(784,hidden_units),
                           nn.functional.Relu(),
                           nn.functional.Linear(hidden_units, 10),
                           ])
    valid = train(model, lr=learning_rate, weight_decay=weight_decay, hidden_units = hidden_units)
    return valid


if __name__ == "__main__":
    a=execute()
    print(a)