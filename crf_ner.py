from crf import *
from crf_data import *
import torch
import numpy as np


crfDat = CrfData("./dataset/train.txt")
log_likelihood = crfDat.log_likelihood()

n_obs = 16
train_x, train_y = crfDat.get_train_data(n_obs)
rolls = np.array(train_x).astype(int)
targets = np.array(train_y).astype(int)

model = CRF(len(crfDat.tag2id), log_likelihood)
model = crf_train_loop(model, rolls, targets, 1, 0.001)

test_roll = np.array(train_x[-1]).astype(int)
test_target = np.array(train_y[-1]).astype(int)

print('test_roll\n', test_roll)
print('model.forward(test_roll)[0]\n', model.forward(test_roll)[0])
print('test_target\n', test_target)
print('list(model.parameters())[0].data.numpy()\n', list(model.parameters())[0].data.numpy())
