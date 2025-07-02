import numpy as np
import math
import time
import matplotlib.pyplot as plt

# Functions for objective (sum-rate) calculation
def obj_IA_sum_rate(H, p, var_noise, K):
    y = 0.0
    for i in range(K):
        s = var_noise
        for j in range(K):
            if j!=i:
                s = s+H[i,j]**2*p[j]
        y = y+math.log2(1+H[i,i]**2*p[i]/s)
    return y

# Functions for WMMSE algorithm
def WMMSE_sum_rate(p_int, H, Pmax, var_noise):
    K = np.size(p_int)
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros(K)
    w = np.zeros(K)
    for i in range(K):
        f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + var_noise)
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + math.log2(w[i])

    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / sum(w * np.square(f) * np.square(H[:, i]))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(K):
            f[i] = H[i, i] * b[i] / ((np.square(H[i, :])) @ (np.square(b)) + var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
            vnew = vnew + math.log2(w[i])

        VV[iter] = vnew
        if vnew - vold <= 2e-2:
            break

    p_opt = np.square(b)
    return p_opt

# Functions for performance evaluation
def perf_eval(H, Py_p, NN_p, K, var_noise=1):
    num_sample = H.shape[2]
    pyrate = np.zeros(num_sample)
    nnrate = np.zeros(num_sample)
    mprate = np.zeros(num_sample)
    rdrate = np.zeros(num_sample)
    for i in range(num_sample):
        pyrate[i] = obj_IA_sum_rate(H[:, :, i], Py_p[:, i], var_noise, K)-0.1
        nnrate[i] = obj_IA_sum_rate(H[:, :, i], NN_p[i, :], var_noise, K)
        mprate[i] = obj_IA_sum_rate(H[:, :, i], np.ones(K), var_noise, K)
        rdrate[i] = obj_IA_sum_rate(H[:, :, i], np.random.rand(K,1), var_noise, K)
    print('Sum-rate: WMMSE: %0.3f, DNN: %0.3f, Max Power: %0.3f, Random Power: %0.3f'%(sum(pyrate)/num_sample, sum(nnrate)/num_sample, sum(mprate)/num_sample, sum(rdrate)/num_sample))
    print('Ratio: DNN: %0.3f%%, Max Power: %0.3f%%, Random Power: %0.3f%%\n' % (sum(nnrate) / sum(pyrate)* 100, sum(mprate) / sum(pyrate) * 100, sum(rdrate) / sum(pyrate) * 100))

    plt.figure('%d'%K)
    plt.style.use('seaborn-deep')
    data = np.vstack([pyrate, nnrate, mprate, rdrate]).T
    #pyrate_data = np.sort(pyrate)
    sorted_data = np.sort(data, axis=0)
    #bins = np.linspace(0, max(pyrate), 50)
    cumulative_percentiles = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    linestyle = ['-.',':','--','-']
    label = ['WMMSE', 'DNN', 'MAX Power', 'Random Power']
    for i in range(len(linestyle)):
        plt.plot(sorted_data[:, i], cumulative_percentiles, linestyle=linestyle[i], label=label[i])

    plt.legend(loc='upper right')
    plt.xlim([0, 8])
    plt.xlabel('sum-rate (bit/sec)', fontsize = 14)
    plt.ylabel('cumulative percentiles', fontsize = 14)
    plt.savefig('fig5_ch5_CDF', format='jpg', dpi=1000)
    plt.show()
    return 0

def perf_eval_H(H, Py_p, NN_p, K, var_noise=1):
    num_sample = H.shape[2]
    pyrate = np.zeros(num_sample)
    nnrate = np.zeros(num_sample)
    mprate = np.zeros(num_sample)
    rdrate = np.zeros(num_sample)
    for i in range(num_sample):
        pyrate[i] = obj_IA_sum_rate(H[:, :, i], Py_p[:, i], var_noise, K)-0.1
        nnrate[i] = obj_IA_sum_rate(H[:, :, i], NN_p[i, :], var_noise, K)
        mprate[i] = obj_IA_sum_rate(H[:, :, i], np.ones(K), var_noise, K)
        rdrate[i] = obj_IA_sum_rate(H[:, :, i], np.random.rand(K,1), var_noise, K)
    np.save('./data/' + 'pyrate.npy', pyrate)
    np.save('./data/' + 'nnrate.npy', nnrate)
    np.save('./data/' + 'mprate.npy', mprate)
    np.save('./data/' + 'rdrate.npy', rdrate)
    print('Sum-rate: WMMSE: %0.3f, DNN: %0.3f, Max Power: %0.3f, Random Power: %0.3f'%(sum(pyrate)/num_sample, sum(nnrate)/num_sample, sum(mprate)/num_sample, sum(rdrate)/num_sample))
    print('Ratio: DNN: %0.3f%%, Max Power: %0.3f%%, Random Power: %0.3f%%\n' % (sum(nnrate) / sum(pyrate)* 100, sum(mprate) / sum(pyrate) * 100, sum(rdrate) / sum(pyrate) * 100))

    plt.figure('%d'%K)
    plt.style.use('seaborn-deep')
    data = np.vstack([pyrate, nnrate]).T
    bins = np.linspace(0, max(pyrate), 50)
    linestyle = ['-','--']
    label = ['WMMSE', 'DNN']
    markers = ['o', 's']
    # for i in range(len(linestyle)):
    #     plt.hist(data[:, i], bins, label=label[i], alpha=0.7, histtype='bar')
    plt.hist(data, bins, alpha=0.7, label=['WMMSE', 'DNN'],)
    plt.legend(loc='upper right')
    plt.xlim([0, 8])
    plt.xlabel('sum-rate (bit/sec x Hz)', fontsize=14)
    plt.ylabel('number of samples', fontsize=14)
    plt.savefig('Histogram_%d.eps'%K, format='eps', dpi=1000)
    plt.show()
    return 0

# Functions for data generation, Gaussian IC case
def generate_Gaussian(K, num_H, Pmax=1, Pmin=0, seed=2017):
    print('Generate Data ... (seed = %d)' % seed)
    np.random.seed(seed)
    Pini = Pmax*np.ones(K)
    var_noise = 1
    X=np.zeros((K**2,num_H))
    Y=np.zeros((K,num_H))
    total_time = 0.0
    for loop in range(num_H):
        CH = 1/np.sqrt(2)*(np.random.randn(K,K)+1j*np.random.randn(K,K))
        H=abs(CH)
        X[:,loop] = np.reshape(H, (K**2,), order="F")
        H=np.reshape(X[:,loop], (K,K), order="F")
        mid_time = time.time()
        Y[:,loop] = WMMSE_sum_rate(Pini, H, Pmax, var_noise)
        total_time = total_time + time.time() - mid_time
    # print("wmmse time: %0.2f s" % total_time)
    return X, Y, total_time

# Functions for data generation, Gaussian IC half user case
def generate_Gaussian_half(K, num_H, Pmax=1, Pmin=0, seed=2017):
    print('Generate Testing Data ... (seed = %d)' % seed)
    np.random.seed(seed)
    Pini = Pmax * np.ones(K)
    var_noise = 1
    X = np.zeros((K ** 2 * 4, num_H))
    Y = np.zeros((K * 2, num_H))
    total_time = 0.0
    for loop in range(num_H):
        CH = 1 / np.sqrt(2) * (np.random.randn(K, K) + 1j * np.random.randn(K, K))
        H = abs(CH)
        mid_time = time.time()
        Y[0: K, loop] = WMMSE_sum_rate(Pini, H, Pmax, var_noise)
        total_time = total_time + time.time() - mid_time
        OH = np.zeros((K * 2, K * 2))
        OH[0: K, 0:K] = H
        X[:, loop] = np.reshape(OH, (4 * K ** 2,), order="F")

    # print("wmmse time: %0.2f s" % total_time)
    return X, Y, total_time

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
import scipy.io as sio
import os
import math
class DNN(nn.Module):
    def __init__(self, n_input, h_hidden1, h_hidden2, h_hidden3, n_output):
        super(DNN, self).__init__()
        self.fc_1 = nn.Linear(n_input, h_hidden1)
        self.fc_1.weight.data.normal_(0, 0.1)
        self.fc_2 = nn.Linear(h_hidden1, h_hidden2)
        self.fc_2.weight.data.normal_(0, 0.1)
        self.fc_3 = nn.Linear(h_hidden2, h_hidden3)
        self.fc_3.weight.data.normal_(0, 0.1)
        self.fc_4 = nn.Linear(h_hidden3, n_output)
        self.fc_4.weight.data.normal_(0, 0.1)

    def forward(self, x, input_keep_prob=1, hidden_keep_prob=1):
        # m = nn.Dropout(1-input_keep_prob)
        # n = nn.Dropout(1-hidden_keep_prob)
        x = F.relu(self.fc_1((x)))
        x = F.relu(self.fc_2((x)))
        x = F.relu(self.fc_3((x)))
        output = F.sigmoid(self.fc_4((x)))
        return output

class PowerControl:
    def __init__(self, X, Y, traintestsplit=0.01, n_hidden_1=200, n_hidden_2=80, n_hidden_3=80, LR=0.0001):
        self.num_total = X.shape[1]  # number of total samples
        self.num_val = int(self.num_total * traintestsplit)  # number of validation samples
        self.num_train = self.num_total - self.num_val  # number of training samples
        self.X_train = torch.tensor(np.transpose(X[:, 0:self.num_train]), dtype=torch.float32)  # training data
        self.Y_train = torch.tensor(np.transpose(Y[:, 0:self.num_train]), dtype=torch.float32)  # training label
        self.X_val = torch.tensor(np.transpose(X[:, self.num_train:self.num_total]), dtype=torch.float32)  # validation data
        self.Y_val = torch.tensor(np.transpose(Y[:, self.num_train:self.num_total]), dtype=torch.float32)  # validation label
        self.n_input = X.shape[0]  # input size
        self.n_output = Y.shape[0]  # output size
        self.lr = LR
        self.DNNs = []
        self.DNNpara = list()
        for i in range(5):
            self.DNN = DNN(self.n_input, n_hidden_1, n_hidden_2, n_hidden_3, self.n_output)
            self.DNNs.append(self.DNN)
            self.DNNpara += list(self.DNN.parameters())
        self.optimizer = torch.optim.RMSprop(self.DNNpara, lr=self.lr)
        self.lamda = 1.0
    def train(self, location, training_epochs=300, batch_size= 100,  LRdecay=0):
        input_keep_prob = 1
        hidden_keep_prob = 1
        total_batch = int(self.num_total/ batch_size)
        start_time = time.time()
        MSETime = np.zeros((training_epochs, 3))
        for i in range(len(self.DNNs)):
            self.DNNs[i].train()
        for epoch in range(1200):
            for i in range(total_batch):
                idx = np.random.randint(self.num_train, size=batch_size)
                for i in range(len(self.DNNs)):

                    '''
                        Write your code here
                    '''

            MSETime[epoch, 0] = np.asarray(self.loss.item())
            # MSETime[epoch, 1] = np.asarray(-(self.sum_rate(self.X_val.detach(), self.DNN(self.X_val).detach(), self.num_val)).item())
            MSETime[epoch, 2] = np.asarray(time.time() - start_time)
            if epoch%(10)==0:
                print('epoch:%d, '%epoch, 'train:%0.2f%%, '%(self.loss*100))
        print("training time: %0.2f s" % (time.time() - start_time))

        sio.savemat('MSETime_qos%d_%d_%d' % (self.n_output, batch_size, self.lr * 10000),
                    {'train': MSETime[:, 0], 'validation': MSETime[:, 1], 'time': MSETime[:, 2]})
        # torch.save(self.DNN.state_dict(), location)
        return 0

    def test(self, H, X, save_name, model_path, binary=0):
        # self.DNN.load_state_dict(torch.load(model_path))
        for i in range(len(self.DNNs)):
            self.DNNs[i].eval()
        X = torch.tensor(np.transpose(X), dtype=torch.float32)
        start_time = time.time()
        num_sample = H.shape[2]
        nnrate = np.zeros(num_sample)
        for j in range(num_sample):
            rate = []
            for i in range(len(self.DNNs)):
                pred = self.DNNs[i](X).detach()[j, :]
                rate.append(obj_IA_sum_rate(H[:, :, j], pred, 1, H.shape[0]))
            nnrate[j] = max(rate)
        return nnrate

    def sum_rate(self, H, P, num_sample, K=10, var_noise=1):
        H = torch.transpose(H, 0, 1)
        H = torch.tensor(np.reshape(H, (K, K, H.shape[1]), order="F"), dtype=torch.float32).clone().detach()
        nnrate = torch.zeros(num_sample)
        nnrate = self.obj_IA_sum_rate(H, P, var_noise, K, num_sample)
        return torch.mean(nnrate)

    def obj_IA_sum_rate(self, H, p, var_noise, K, num_sample):
        y = torch.zeros(num_sample)
        r = torch.zeros((num_sample, K))
        r_min = torch.full((num_sample, K), 1.0)
        for i in range(K):
            s = var_noise
            for j in range(K):
                if j != i:
                    s = s + H[i, j, :] ** 2 * p[:, j]
            y = y + torch.log2(1 + H[i, i, :] ** 2 * p[:, i] / s)
            r[:, i] = F.relu(r_min[:, i] - torch.log2(1 + H[i, i, :] ** 2 * p[:, i] / s))
        r = self.lamda * torch.sum(r, dim=1)
        return y-r

K = 10                     # number of users
num_H = 10000              # number of training samples
num_test = 500            # number of testing  samples
training_epochs = 1000     # number of training epochs
trainseed = 0              # set random seed for training set
testseed = 7               # set random seed for test set

# Problem Setup
print('Gaussian IC Case: K=%d, Total Samples: %d, Total Iterations: %d\n'%(K, num_H, training_epochs))

# Generate Training Data
Xtrain, Ytrain, wtime = generate_Gaussian(K, num_H, seed=trainseed)
DNN = PowerControl(X=Xtrain, Y=Ytrain)

# Training Deep Neural Networks
#print('train DNN ...')
# Save & Load model from this path
model_location = "./DNNmodel/model_demok_qos=%d.ckpt"%K
DNN.train(model_location, training_epochs=training_epochs, batch_size=1000,  LRdecay=0)

# Generate Testing Data
X, Y, wmmsetime = generate_Gaussian(K, num_test, seed=testseed)

# Testing Deep Neural Networks

# print('wmmse time: %0.3f s, dnn time: %0.3f s, time speed up: %0.1f X' % (wmmsetime, dnntime, wmmsetime / dnntime))

# Evaluate Performance of DNN and WMMSE
H = np.reshape(X, (K, K, X.shape[1]), order="F")
nnrate = DNN.test(H, X, "Prediction_qos%d" % K, model_location, binary=0)
# NNVbb = sio.loadmat('Prediction_qos%d' % K)['pred']
perf_eval(H, Y, nnrate, K)