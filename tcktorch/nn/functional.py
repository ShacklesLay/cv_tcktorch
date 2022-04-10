import numpy as np

class Layer:
    def forward(self,x):
        raise NotImplementedError

    def backward(self, dz):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

class Flatten(Layer):

    def forward(self, inputs):
        self.shape = inputs.shape
        a = inputs.reshape((self.shape[0], -1))
        return a

    def backward(self, da):
        return da.reshape(self.shape)

class Linear(Layer):
    def __init__(self,input_dim, output_dim, bias=True):
        self.parameters = {}
        self.grads = {}
        self.bias = bias
        self.parameters['W'] = np.random.randn(input_dim,output_dim)*0.01
        if bias:
            self.parameters['b'] = np.zeros((1, output_dim))

    def forward(self,x):
        '''

        :param x: (#datas, input_dim)
        :return:
        '''
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

class Finetuning(Layer):
    def __init__(self,input_dim, output_dim, bias=True):
        self.parameters = {}
        self.grads = {}
        self.bias = bias
        self.parameters['W'] = np.random.randn(input_dim,output_dim)*0.01
        if bias:
            self.parameters['b'] = np.zeros((1, output_dim))

    def forward(self,x):
        return x, self.parameters


# def padding(x,pad):
#     x_pad= np.pad(x,((0, 0),(pad, pad),(pad, pad),(0, 0)), 'constant', constant_values=0)
#     return x_pad
#
# def conv_single_step(a_slice_prev,W,b):
#     s = np.multiply(a_slice_prev, W) + b
#     Z = np.sum(s)
#     return Z
#
# class Conv2d(Layer):
#     def __init__(self, kernel_size, channels_in, channels_out,stride, pad):
#         self.kernel_size = kernel_size
#         self.parameters = {}
#         self.parameters['W'] = np.random.randn(kernel_size[0],kernel_size[1], channels_in, channels_out)
#         self.parameters['b'] = np.zeros((1,1,1,channels_out))
#         self.stride = stride
#         self.pad = pad
#         self.grads={}
#
#     def forward(self,x):
#         m, H_prev, W_prev, C_prev = x.shape
#         f, f, C_prev, C = self.parameters['W'].shape
#         #计算输出的维度
#         H = 1+ int((H_prev+2*self.pad-f)/self.stride)
#         W = 1+ int((W_prev+2*self.pad-f)/self.stride)
#         #初始化输出
#         Z = np.zeros((m, H, W, C))
#         #零扩展
#         A_prev_pad = padding(x, self.pad)
#
#         for i in range(m):
#             a_prev_pad = A_prev_pad[i]
#             for h in range(H):
#                 for w in range(W):
#                     for c in range(C):
#                         #找到当前'slice'的角坐标
#                         vert_start = h * self.stride
#                         vert_end = vert_start + f
#                         horiz_start = w * self.stride
#                         horiz_end = horiz_start + f
#                         #找到slice
#                         a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
#                         #卷积
#                         Z[i, h, w, c] = conv_single_step(a_slice_prev, self.parameters['W'][:, :, :, c], self.parameters['b'][:, :, :, c])
#         self.cache = [x]
#         return Z
#
#     def backward(self, dz):
#         A_prev = self.cache[0]
#         m, H_prev, W_prev, C_prev = A_prev.shape
#         f, f, C_prev, C = self.parameters['W'].shape
#         m,H,W,C = dz.shape
#
#         #初始化
#         dA_prev = np.zeros((m, H_prev, W_prev, C_prev))
#         self.grads['dW'] = np.zeros((f, f, C_prev, C))
#         self.grads['db'] = np.zeros((1, 1, 1, C))
#
#         A_prev_pad = padding(A_prev, self.pad)
#         dA_prev_pad = padding(dA_prev, self.pad)
#
#         for i in range(m):
#             a_prev_pad = A_prev_pad[i]
#             da_prev_pad = dA_prev_pad[i]
#             for h in range(H):
#                 for w in range(W):
#                     for c in range(C):
#                         vert_start = h * self.stride
#                         vert_end = vert_start + f
#                         horiz_start = w * self.stride
#                         horiz_end = horiz_start + f
#                         a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
#
#                         da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.parameters['W'][:, :, :, c] * dz[i, h, w, c]
#                         self.grads['dW'][:,:,:,c] += a_slice * dz[i, h, w, c]
#                         self.grads['db'][:,:,:,c] += dz[i, h, w, c]
#             dA_prev[i, :, :, :] = dA_prev_pad[i, self.pad:-self.pad, self.pad:-self.pad, :]
#         return dA_prev

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


class Relu(Layer):
    def forward(self,z):
        self.cache = [z]
        a= z * (z>0)
        return a

    def backward(self, da):
        z = self.cache[0]
        dz = da * (z>0)
        return dz

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

class Dropout(Layer):
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, a):
        self.mask = np.random.rand(1,a.shape[1]) < self.p
        return a*self.mask/self.p

    def backward(self, da):
        return da*self.mask/self.p


class Result:
    def __init__(self, grad, loss, nnModel):
        self.grad = grad
        self.loss = loss
        self.nnModel = nnModel

    def backward(self,tuning=False):
        if tuning:
            self.nnModel.backward(self.grad, tuning)
        else:
            self.nnModel.backward(self.grad)

class LossFunction:
    def __init__(self, nnModel):
        '''

        :param nnModel: Sequential.py 中的Module类
        '''
        self.nn = nnModel

    def loss_compute(self, input, target):
        raise NotImplementedError

    def __call__(self, input, target):
        '''

        :param input:
        :param target: one-hot array
        :return:
        '''
        return self.loss_compute(input, target)


class MseLoss(LossFunction):
    def loss_compute(self, input, target):
        grad = 2*(input - target)
        loss = np.mean((target-input)**2)
        return Result(grad, loss, self.nn)


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






