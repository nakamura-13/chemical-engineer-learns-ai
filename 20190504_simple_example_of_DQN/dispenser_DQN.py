# -*- coding: utf-8 -*-
"""
DQNの場合
"""

class Dispenser(object):
    def __init__(self, init_state):
        """
        初期のON/OFF状態を設定する
        init_state: 0->電源OFF、1->電源ON
        """
        self.state  = init_state

    def powerbutton(self):
        """
        電源ボタンを押し、ON/OFFを切り替える
        """
        if self.state == 0:
            self.state = 1
        else:
            self.state = 0

    def step(self, action):
        """
        払出機を操作する
        action: 0->電源ボタンを押す 1->払出ボタンを押す
        状態と報酬が返る
        """
        if action == 0: # 電源ボタンを押した場合
            self.powerbutton() # 電源ON/OFF切り替え
            reward = 0 # 報酬はない
        else:           # 払出ボタンを押した場合
            if self.state == 1:
                reward = 1 # 電源ONのときのみ報酬あり
            else:
                reward = 0
        return self.state, reward

############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(1, 3)
        self.l2 = nn.Linear(3, 3)
        self.l3 = nn.Linear(3, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

dqn = DQN()
optimizer = torch.optim.SGD(dqn.parameters(), lr=0.01)
criterion = nn.MSELoss()

############################################################################

def update_dqn(state, action, next_state, reward):
    ## 各変数をtensorに変換
    state = torch.FloatTensor([state])
    action = torch.LongTensor([action]) # indexとして使うのでLong型
    next_state = torch.FloatTensor([next_state])

    ## Q値の算出
    q_now = dqn(state).gather(-1, action) # 今の状態のQ値
    max_q_next = dqn(next_state).max(-1)[0].detach() # 状態移行後の最大のQ値
    gamma = 0.9
    q_target = reward + gamma * max_q_next # 目標のQ値

    # DQNパラメータ更新
    optimizer.zero_grad()
    loss = criterion(q_now, q_target) # 今のQ値と目標のQ値で誤差を取る
    loss.backward()
    optimizer.step()
    
    return loss.item() # lossの確認のために返す

###########################################################################

import numpy as np

EPS_START = 0.9
EPS_END = 0.0
EPS_DECAY = 200

def decide_action(state, episode):
    state = torch.FloatTensor([state]) # 状態を1次元tensorに変換
    ## ε-グリーディー法
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-episode / EPS_DECAY)
    if eps <= np.random.uniform(0, 1):
        with torch.no_grad():
            action = dqn(state).max(-1)[1] # Q値が最大のindexが得られる
            action = action.item() # 0次元tensorを通常の数値に変換
    else:
        num_actions = len(dqn(state)) # action数取得
        action = np.random.choice(np.arange(num_actions)) # ランダム行動
    return action

###########################################################################

import matplotlib.pyplot as plt

NUM_EPISODES = 1200
NUM_STEPS = 5

log = [] # 結果のプロット用

for episode in range(NUM_EPISODES):
    env = Dispenser(0)
    total_reward = 0 # 1エピソードでの報酬の合計を保持する

    for s in range(NUM_STEPS):
        ## 現在の状態を確認
        state = env.state
        ## 行動を決める
        action = decide_action(state, episode)
        ## 決めた行動に従いステップを進める。次の状態、報酬を得る
        next_state, reward = env.step(action)
        ## DQNを更新
        loss = update_dqn(state, action, next_state, reward)
        total_reward += reward

    log.append([total_reward, loss])

r, l = np.array(log).T
fig = plt.figure(figsize=(11,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Loss")
ax1.set_yscale("log")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Total reward")
ax1.plot(l)
ax2.plot(r)
plt.show()
