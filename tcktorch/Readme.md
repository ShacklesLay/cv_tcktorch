# Readme

## 文件说明：

仿照pytorch，基于numpt，自定义了tcktorch模块，能够实现基本的神经网络训练功能。



1. tcktorch文件夹：自定义神经网络训练模块
   1. nn文件夹：包含fuctional.py 和 Sequential.py文件
      1. functional.py文件：实现相关层的函数，包括线性层、激活函数、损失函数等
      2. Sequential.py文件：仿照pytorch定义了 Sequential 容器，用于实现隐藏层自动的前向和后向传播
   2. utils文件夹：包含 data.py文件
      1. data.py文件：仿照pytorch定义了 Dataset 类和 Dataloader 类
   3. torchvision文件夹：包含Transform.py文件
      1. Transform.py文件：实现了数据增强的基本操作
   4. optim.py文件：实现了带动量的SGD




## 代码说明：

#### 1.1.1 functional.py文件 

##### 父类 Layer

在每一层中，我们需要实现前向传播，求导和后向传播三项操作。

```python
class Layer:

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dz):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)
```

##### Flatten函数

forward 将矩阵拉成向量， backward 将形状复原

```python
class Flatten(Layer):

    def forward(self, inputs):
        self.shape = inputs.shape
        a = inputs.reshape((self.shape[0], -1))
        return a

    def backward(self, da):
        return da.reshape(self.shape)
```



##### 卷积层

```python
class Conv2d(Layer):
    def __init__(self, kernel_size, in_channels, out_channels, padding=0, stride=1, bias=True):
        self.parameters = {}
        #glorot = np.sqrt(2 / ((in_channels + out_channels) * kernel_size[0] * kernel_size[1]))
        self.parameters['kernel'] = np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]) * \
                                    np.sqrt(2 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.bias = bias
        if self.bias:
            self.parameters['b'] = np.zeros((1, 1, 1, out_channels))
        self.padding = padding
        self.stride = stride
        self.grads = {}

    def forward(self, inputs):
        x = inputs.copy()
        outputs, stack = self.__conv(x=x, kernel=self.parameters['kernel'], padding=self.padding, stride=self.stride)
        self.cache = [stack]
        if self.bias:
            outputs += self.parameters['b']
        return outputs

    @classmethod
    def __conv(cls, x, kernel, padding, stride):  # x: m * height * width * channel
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), constant_values=0)
        m, xh, xw = x.shape[:3]
        x = np.transpose(x, (0, 3, 1, 2))
        out_channels, in_channels, kh, kw = kernel.shape
        line_kernel = kernel.reshape((out_channels, in_channels, kh * kw, 1))
        line_kernel = np.expand_dims(line_kernel, 1).repeat(m, axis=1)
        out_height = (xh - kh) // stride + 1
        out_width = (xw - kw) // stride + 1
        stack = cls.img2col(x, kw, kh, stride)
        outputs = np.transpose(np.sum(stack @ line_kernel, axis=2).reshape((out_channels, m, out_height, out_width)),
                               (1, 2, 3, 0))
        return outputs, stack  # outputs: m * height * width * channels

    @staticmethod
    def img2col(img, kw, kh, stride):
        m, in_channels, xh, xw = img.shape
        flag = True
        for h in range(0, xh + 1 - kh, stride):
            for w in range(0, xw + 1 - kw, stride):
                tmp = img[:, :, h:(h + kh), w:(w + kw)].reshape(m, in_channels, 1, kh * kw)
                if flag:
                    stack = tmp
                    flag = False
                else:
                    stack = np.concatenate((stack, tmp), axis=2)
        return stack

    @staticmethod
    def stride_padding(inputs, stride):  # inputs: m, channels, height, width
        m, channels, height, width = inputs.shape
        padn = stride - 1
        paded = np.append(inputs.reshape((m, channels, height, width, 1)), np.zeros((m, channels, height, width, padn)),
                          axis=4).reshape((m, channels, height, width * (padn + 1)))[:, :, :, :-padn]
        m, channels, height, width = paded.shape
        paded = np.append(paded.transpose(0, 1, 3, 2).reshape((m, channels, width, height, 1)),
                          np.zeros((m, channels, width, height, padn)), axis=4).reshape(
            (m, channels, width, height * (padn + 1)))[:, :, :, :-padn].transpose(0, 3, 2, 1)
        return paded  # inputs: m, height, width, channels

    def backward(self, dz, backward=True):  # dz m * height * width * out_channels
        if self.bias:
            self.grads['b'] = np.mean(dz, axis=(0, 1, 2))
        m, height, width, out_channels = dz.shape
        in_channels, kh, kw = self.parameters['kernel'].shape[1:]
        ds = np.transpose(dz, (3, 0, 1, 2))
        ds = ds.reshape((out_channels, m, height * width, 1))
        ds = np.expand_dims(ds, 2).repeat(in_channels, axis=2) # out_channels * m * in_channels * (height * width) * 1
        stack = self.cache[0]  # m * in_channels * (height * width)  * (kh * kw)
        dw = (np.transpose(stack, (0, 1, 3, 2)) @ ds).reshape((out_channels, m, in_channels, kh, kw))
        self.grads['kernel'] = np.mean(dw, axis=1)
        if backward:
            if self.stride != 1:
                dz = np.transpose(dz, (0, 3, 1, 2))  # m * out_channels * height * width
                dz = self.stride_padding(dz, self.stride)
            kernel = self.parameters['kernel'][:, :, ::-1, ::-1].transpose(1, 0, 2, 3)
            padding = kernel.shape[2] - self.padding - 1
            dz, _ = self.__conv(x=dz, kernel=kernel, padding=padding, stride=1)
            return dz
```



##### 线性层

forward: $$z = xW+b$$   

backward: $$dw = \frac{1}{m}x^T dz$$    $$db = \frac{1}{m}dz$$	$$dx=dzW^T$$

```python
class Linear(Layer):
    def __init__(self,input_dim, output_dim, bias=True):
        self.parameters = {}
        self.grads = {}
        self.bias = bias
        self.parameters['W'] = np.random.randn(input_dim,output_dim)*0.01
        if bias:
            self.parameters['b'] = np.zeros((1, output_dim))

    def forward(self,x):
        self.cache = [x]
        if self.bias:
            z = np.matmul(x, self.parameters['W']) + self.parameters['b']
        else:
            z = np.matmul(x, self.parameters['W'])
        return z

    def backward(self, dz):
        x = self.cache[0]
        self.grads['dW'] = np.matmul(x.T, dz)/x.shape[1]
        if self.bias:
            self.grads['db'] = np.sum(dz)/x.shape[1]
        return np.matmul(dz, self.parameters['W'].T)
```

##### 激活函数

###### Relu

forward: $$a=max(z,0)$$

backward: $$dz =1 \quad if\quad a>0\quad else\quad 0 $$

```python
class Relu(Layer):
    def forward(self,z):
        self.cache = [z]
        a= z * (z>0)
        return a

    def backward(self, da):
        z = self.cache[0]
        dz = da * (z>0)
        return dz
```

###### Tanh

forward: $$a=\frac{e^z-e^{-z}}{e^z+e{-z}}$$

backward: $$dz=da(1-a^2)$$

```python
class Tanh(Layer):

    def forward(self, z):
        ez = np.exp(z)
        nez = np.exp(-z)
        a = (ez - nez) / (ez + nez)
        self.cache = [a]
        return a

    def backward(self, da):
        a = self.cache[0]
        return da * (1 - a ** 2)
```

##### Dropout

forward: $a = z*mask/p$

backward: $dz =da*mask/p $

```python
class Dropout(Layer):
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, a):
        self.mask = np.random.rand(1,a.shape[1]) < self.p
        return a*self.mask/self.p

    def backward(self, da):
        return da*self.mask/self.p
```



##### 损失函数及调用

###### Lossfunction	损失函数父类

```python
class LossFunction:
    def __init__(self, nnModel):
        self.nn = nnModel

    def loss_compute(self, input, target):
        raise NotImplementedError

    def __call__(self, input, target):
        return self.loss_compute(input, target)
```

###### Result 损失函数返回类

调用损失函数的返回值是一个类，属性包含隐层容器，损失和梯度返回值，并能由此执行backward

```python
class Result:
    def __init__(self, grad, loss, nnModel):
        self.grad = grad
        self.loss = loss
        self.nnModel = nnModel

    def backward(self):
        self.nnModel.backward(self.grad)
```

###### MSELoss

```python
class MseLoss(LossFunction):
    def loss_compute(self, input, target):
        grad = 2*(input - target)
        loss = np.mean((target-input)**2)
        return Result(grad, loss, self.nn)

```

$$
loss = \frac{\sum_{i=0}^m\sum_{j=0}^{class\_num}(input_{i,j}-target_{i,j})}{m}
$$

$$
grad = 2(input-target)/m
$$

###### CrossEntropyLoss

定义了交叉熵损失函数，并且定义了softmax函数，作为使用CrossEntropyLoss时，输出层的激活函数

```python
class CrossEntropyLoss(LossFunction):
    def softmax(self, z):
        ez = np.exp(z)
        a = ez/np.sum(ez,axis=1).reshape(ez.shape[0],1)
        return a

    def loss_compute(self, input, target):
        yhat = self.softmax(input)
        grad = yhat-target
        loss = -np.sum(target*np.log(yhat))/yhat.shape[0]
        return Result(grad, loss, self.nn)

```



#### 1.1.2 Sequential.py文件

定义了父类Modules，可实现自动的前向和后向传播。

实现了保存模型参数的函数 save_state_dic 和加载模型参数的函数 load_state_dic 

```python
class Modules:
    def forward(self, x, eval_pattern):
        for layer in self.layers.values():
            if eval_pattern and layer.__class__.__name__ == 'Dropout':
                continue
            x = layer(x)
        return x

    def backward(self, grad,tuning=False):
        for layer in reversed(self.layers.values()):
            if tuning:
                if layer == list(self.layers.values())[-1]:
                    grad = layer.backward(grad)
                else:
                    grad = layer.backward(np.zeros_like(grad))
            else:
                grad = layer.backward(grad)

    def __call__(self, x, eval_pattern=False):
        return self.forward(x, eval_pattern=eval_pattern)

    def save_state_dict(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.parameters, f, pickle.HIGHEST_PROTOCOL)
        return

    def load_state_dict(self, state_dict=None, path=None):
        if not state_dict:
            import pickle
            with open(path, 'rb') as f:
                state_dict = pickle.load(f)
        for layer in state_dict.keys():
            for param, v in state_dict[layer].items():
                try:
                    self.parameters[layer][param] = v
                except:
                    continue
        return
```

定义Sequential类，整合隐藏层和隐藏层的参数

```python
class Sequential(Modules):
    def __init__(self, layers):
        '''
        :param layers: nn.functional里的layer类
        '''
        self.layers = {}
        self.parameters = {}
        self.name_cout = {}
        for layer in layers:
            name = layer.__class__.__name__
            try:
                self.name_cout[name] +=1
            except:
                self.name_cout[name] =1
            self.layers[name.lower() + str(self.name_cout[name])] = layer
            try:
                self.parameters[name.lower() + str(self.name_cout[name])] = layer.parameters
            except:
                continue
```



#### 1.2.1 data.py文件

###### Dataset父类

```python
class Dataset:

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
```

###### Dataloader

实现Dataloader的基本功能，按照batch_size取出Dataset的数据

```python
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
```

#### 1.3 torchvision文件夹

##### 1.3.1 Transform.py文件

基于双线性插值实现了 resize, translation, totation

```python
import numpy as np


def BilinearInterpolation(img, pos):
    A = img.transpose(2, 0, 1)
    x, y = pos
    x0, x1 = int(x), int(x + 1)
    y0, y1 = int(y), int(y + 1)
    if x == x0 and y == y0:
        return A[:, x0:x1, y0:y1].transpose(1, 2, 0)
    return (np.array([x1 - x, x - x0]).reshape((1, 2)) @ A[:, x0:(x1 + 1), y0:(y1 + 1)] @
            np.array([y1 - y, y - y0]).reshape((2, 1))).transpose(1, 2, 0)


class Resize:

    def __init__(self, ran=(0.75, 1.0)):
        self.l, self.u = ran

    def __call__(self, image):
        input_shape = image.shape
        H, W, C = image.shape
        H_ratio = np.random.uniform(self.l, self.u)
        W_ratio = np.random.uniform(self.l, self.u)
        outH, outW = int(H*H_ratio), int(W*W_ratio)
        output = np.zeros((outH, outW, C))
        count = 0
        for h in range(int(H*H_ratio)):
            for w in range(int(W*W_ratio)):
                count += 1
                output[h:(h+1), w:(w+1), :] = BilinearInterpolation(image, (h/H_ratio, w/W_ratio))
        ph, pw = (H-outH)//2, (W-outW)//2
        output = np.pad(output, ((ph, H-outH-ph), (pw, W-outW-pw), (0, 0)),
                        "constant", constant_values=0)
        return output.reshape(input_shape)


class Translation:

    def __init__(self, distortion=0.1):
        self.distortion = distortion

    def __call__(self, image):
        input_shape = image.shape
        H, W, C = image.shape
        h, w = int(H * self.distortion), int(W * self.distortion)
        output = np.pad(image, ((h, h), (w, w), (0, 0)), "constant", constant_values=0)
        HFLAG, WFLAG = np.random.uniform(size=2) > 0.5
        output = output[2*h*HFLAG: 2*h*HFLAG+H, 2*w*WFLAG: 2*w*WFLAG+W]
        return output.reshape(input_shape)


class Rotation:

    def __init__(self, ran=(-np.pi/12, np.pi/12)):
        self.l, self.u = ran

    def __call__(self, image: np.ndarray) -> np.ndarray:
        input_shape = image.shape
        H, W, C = image.shape
        phi = np.random.uniform(self.l, self.u)
        rotate_matrix = np.array([[ np.cos(phi), np.sin(phi), 0.],
                                  [-np.sin(phi), np.cos(phi), 0.],
                                  [           0,           0, 1.]])
        rotate_matrix = np.linalg.inv(rotate_matrix)
        output = np.zeros_like(image)
        center = np.array([[H/2], [W/2], [0]])
        location = np.array([[h, w, 1.] for h in range(H) for w in range(W)]).T
        target = rotate_matrix @ (location - center) + center
        location = location.astype(np.int)
        for h, w, tx, ty in zip(location[0], location[1], target[0], target[1]):
            if tx >= H-1 or tx < 0 or ty >= W-1 or ty < 0: continue
            output[h, w, :] = BilinearInterpolation(image, (tx, ty))
        return output.reshape(input_shape)
```



#### 1.4 optim.py文件

##### Gradient Descent

实现带动量的梯度下降：

更新公式：
$$
v_o = 0\\
v = momentum*v+(1-momentum)*grad\\
param = param-learning_\_rate*v
$$


```python
class GD:
    def __init__(self, lr, nnModel, momentum=0.9, weight_decay=0):
        self.nn = nnModel
        self.lr = lr
        self.v_dic = {}
        self.momentum = momentum
        self.t = 0
        self.weight_decay = weight_decay
        for name, layer in self.nn.layers.items():
            try:
                self.v_dic[name] = {}
                for parameter in layer.parameters.keys():
                    self.v_dic[name][parameter] = 0
            except:
                continue

    def step(self):
        self.t +=1
        for name, layer in self.nn.layers.items():
            try:
                for parameter in layer.parameters.keys():
                    self.v_dic[name][parameter] = self.momentum*self.v_dic[name][parameter] + (1-self.momentum) * layer.grads['d'+parameter]
                    v_corrcet = self.v_dic[name][parameter]/(1-self.momentum ** self.t)
                    layer.parameters[parameter] =(1-self.weight_decay)*layer.parameters[parameter] - self.lr * v_corrcet

            except:
                continue
```

