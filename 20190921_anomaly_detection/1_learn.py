# -*- coding: utf-8 -*-
"""
異常検知 学習
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

## データ読み込み
f  = np.load("f_pv.npy")
tf = np.load("tf_pv.npy")
p  = np.load("p_pv.npy")
tm = np.load("tm_pv.npy")
to = np.load("to_pv.npy")

## 700日までを学習データとするようデータ編集
x_data = f.reshape(1, -1) # 2次元配列に変換
x_data = np.append(x_data, tf.reshape(1, -1), axis=0) # 2次元配列にしてappend
x_data = np.append(x_data, p.reshape(1, -1), axis=0)
x_data = np.append(x_data, tm.reshape(1, -1), axis=0)
x_data = torch.FloatTensor(x_data.T)
x_train = x_data[1:701] # x_dataの700日目までを学習用データとする

y_data = torch.FloatTensor(to.reshape(1, -1).T)
y_train = y_data[1:701]

dataset = torch.utils.data.TensorDataset(x_train, y_train)

## ニューラルネットワークモデル設定
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(4, 20) # 入力4つ (原料流量、原料温度、反応圧力、熱媒体温度)
        self.l2 = nn.Linear(20, 20)
        self.l3 = nn.Linear(20, 1) # 出力1つ (出口温度)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

## 学習準備
model = Model()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=70, shuffle=True)

## 学習
loss_log = []
epoch = 1000
for e in range(epoch):
    for xt, yt in data_loader:
        optimizer.zero_grad()
        y_model = model(xt)
        loss = criterion(y_model, yt)
        loss.backward()
        optimizer.step()
    loss_log.append(loss.item())
    if e%10==0:
        print(e, loss.item())

y_model = model(x_train)

'''
# ネットワークsave
torch.save(model.state_dict(), "model.pth")

# データプロット用
fig1 = plt.figure(figsize=(12,6))
ax11 = fig1.add_subplot(121)
ax12 = fig1.add_subplot(122)
ax11.plot(loss_log)
ax11.set_yscale("log")
ax11.set_xlabel("Epoch")
ax11.set_ylabel("Loss")
ax12.plot(y_train.numpy().T[0], label="Train")
ax12.plot(y_model.detach().numpy().T[0], label="Model")
ax12.legend()
ax12.set_xlabel("Time on stream, day")
ax12.set_ylabel("Outlet temp., ℃")
plt.show()
'''
