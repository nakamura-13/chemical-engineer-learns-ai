# -*- coding: utf-8 -*-
"""
異常検知 結果確認
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
to = np.load("to_an.npy")

## データ編集
x_data = f.reshape(1, -1) # 2次元配列に変換
x_data = np.append(x_data, tf.reshape(1, -1), axis=0) # 2次元配列にしてappend
x_data = np.append(x_data, p.reshape(1, -1), axis=0)
x_data = np.append(x_data, tm.reshape(1, -1), axis=0)
x_data = torch.FloatTensor(x_data.T)

y_data = torch.FloatTensor(to.reshape(1, -1).T)

## ニューラルネットワークモデル設定
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(4, 20)
        self.l2 = nn.Linear(20, 20)
        self.l3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

## 学習データ読み込み
model = Model()
model.load_state_dict(torch.load("model.pth"))

## モデル値と実測値比較、全データ範囲で
y_model = model(x_data)
deviation = (y_model - y_data).abs()

fig1 = plt.figure(figsize=(8,12))
ax11 = fig1.add_subplot(211)
ax12 = fig1.add_subplot(212)
ax11.plot(y_data.numpy().T[0], label="Data")
ax11.plot(y_model.detach().numpy().T[0], label="Model")
ax11.legend()
ax11.set_xlabel("Time on stream [day]")
ax11.set_ylabel("Outlet temp. [℃]")
ax11.set_xlim(0,1000)
ax11.set_ylim(260,340)
ax11.grid(linestyle='--')
ax12.plot(deviation.detach().numpy().T[0])
ax12.set_xlabel("Time on stream [day]")
ax12.set_ylabel("Deviation [℃]")
ax12.set_xlim(0,1000)
ax12.set_ylim(0, 40)
ax12.grid(linestyle='--')
plt.show()
