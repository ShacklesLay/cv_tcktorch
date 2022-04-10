import numpy as np

class Modules:
    def forward(self, x, eval_pattern):
        for layer in self.layers.values():
            if eval_pattern and layer.__class__.__name__ == 'Dropout':
                continue
            x = layer(x)
        return x

    def backward(self, grad,tuning=False):
        for layer in reversed(list(self.layers.values())):
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
