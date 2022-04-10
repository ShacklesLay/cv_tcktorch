import numpy as np

class GD:
    def __init__(self, lr, nnModel, momentum=0.9, weight_decay=0.00001, decay_rate=0.005):
        self.nn = nnModel
        self.lr = lr
        self.v_dic = {}
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.decay_rate = decay_rate
        for name, layer in self.nn.layers.items():
            try:
                self.v_dic[name] = {}
                for parameter in layer.parameters.keys():
                    self.v_dic[name][parameter] = 0
            except:
                continue

    def step(self, epoch):
        lr = 1/(1+self.decay_rate*(epoch))*self.lr
        for name, layer in self.nn.layers.items():
            try:
                for parameter in layer.parameters.keys():
                    self.v_dic[name][parameter] = self.momentum*self.v_dic[name][parameter] + (1-self.momentum) * layer.grads['d'+parameter]
                    v_corrcet = self.v_dic[name][parameter]/(1-self.momentum ** (epoch+1))
                    layer.parameters[parameter] =(1-self.weight_decay)*layer.parameters[parameter] - lr * v_corrcet

            except:
                continue
