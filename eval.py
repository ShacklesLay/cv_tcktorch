import os
import gzip
from tcktorch.utils.data import Dataset, Dataloader
from tcktorch import nn
from tqdm import tqdm
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

class TestData(Dataset):
    def __init__(self):
        data = (test_images, test_labels)
        x = data[0]/255   #归一化可以加速拟合
        self.x = x.reshape(x.shape[0], 28, 28, 1)
        y = data[1]
        self.y = y.reshape(y.shape[0],1)      #测试集不需要计算损失，所以不用变为one-hot array

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

def accuracy(y_hat, y, one_hot = True):
    y_hat_trans = np.argmax(y_hat,axis=1).reshape(y_hat.shape[0], 1)
    y_trans = y
    if one_hot:
        y_trans = np.argmax(y,axis=1).reshape(y.shape[0], 1)
    return np.sum(y_hat_trans == y_trans)

def test(model,batch_size=64, seed=37):
    np.random.seed(seed)
    testdata = TestData()
    testloder = Dataloader(dataset=testdata, batch_size=batch_size)

    accn = 0
    for x_valid, ytarget_valid in tqdm(testloder):
        y_valid = model(x_valid, eval_pattern=True)
        accn += accuracy(y_valid, ytarget_valid, one_hot=False)

    print("Accurary: {}".format(accn/len(testdata)))


def evalexe(hidden_units: object = 128, learning_rate: object = 0.1, weight_decay: object = 1e-5) -> object:

    # file_object = open('result2.txt')
    # a = file_object.read()
    # learning_rate = float(re.search("(?<=lr:)[0-9\.e-]*", a).group())
    # hidden_units = int(re.search("(?<=its:)[0-9\.]*", a).group())
    # weight_decay = float(re.search("(?<=cay:)[0-9\.e-]*", a).group())

    model = nn.Sequential([nn.functional.Flatten(),
                           nn.functional.Linear(784, hidden_units),
                           nn.functional.Relu(),
                           nn.functional.Linear(hidden_units, 10),
                           ])
    # path = os.path.join('model', 'lr{}hiddenunits_{}weightdecay_{}'.format(learning_rate, hidden_units, weight_decay))
    # path1 = os.path.join(path, 'model.pkl')
    path1 = 'model.pkl'
    model.load_state_dict(path=path1)
    test(model=model)

if __name__ == '__main__':
    evalexe()
