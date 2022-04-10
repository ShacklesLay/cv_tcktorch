from train import *

size=100
hidden_units = np.random.randint(low=300, high=350, size=size)
lr = np.random.rand(size) *(-2)
lr = np.power(10, lr)
weight_decay = np.random.rand(size) * (-3)-5
weight_decay = np.power(10, weight_decay)
hidden_units_list=[]
lr_list = []
weight_decay_list = []
validlist=[]
for i in range(size):
    valid_acc = execute(hidden_units=hidden_units[i],learning_rate=lr[i],weight_decay=weight_decay[i])
    hidden_units_list.append(hidden_units[i])
    lr_list.append(lr[i])
    weight_decay_list.append(weight_decay[i])
    validlist.append(valid_acc)

validlist = np.array(validlist)
a = np.argmax(validlist)

filename = 'result1.txt'
with open(filename, 'w') as file_object:
    file_object.write('Best model: lr:{}\thidden_units:{}\tweight_decay:{}\tvalid_acc:{:4f}'.format(lr_list[a], hidden_units_list[a], weight_decay_list[a],validlist[a]))







