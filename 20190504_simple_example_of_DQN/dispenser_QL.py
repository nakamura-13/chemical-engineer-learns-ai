# -*- coding: utf-8 -*-
"""
Q学習の場合
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

###############################################################################

import numpy as np

qtable = np.zeros((2, 2))

###############################################################################

def update_qtable(qtable, state, action, next_state, reward):
    gamma = 0.9
    alpha = 0.5
    next_qmax = max(qtable[next_state])
    qtable[state, action] = (1 - alpha) * qtable[state, action] + alpha * (
        reward + gamma * next_qmax)
    return qtable

###############################################################################

EPS_START = 0.9
EPS_END = 0.0
EPS_DECAY = 200

def decide_action(qtable, state, episode):
    ## ε-グリーディー法
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-episode / EPS_DECAY)
    if eps <= np.random.uniform(0, 1):
        t = np.where(qtable[state]==qtable[state].max())[0] # Q値が最大のインデックスを返す
    else:
        t = np.arange(qtable.shape[-1]) # 取りうる行動すべて
    return np.random.choice(t) # 行動の候補からランダムに選ぶ

###############################################################################

NUM_EPISODES = 1200
NUM_STEPS = 5

for episode in range(NUM_EPISODES):
    env = Dispenser(0)
    total_reward = 0 # 1エピソードでの報酬の合計を保持する

    for d in range(NUM_STEPS):
        ## 現在の状態を確認
        state = env.state
        ## 行動を決める
        action = decide_action(qtable, state, episode)
        ## 決めた行動に従いステップを進める.また次の状態、報酬を得る
        next_state, reward = env.step(action)
        ## Qテーブル更新
        qtable = update_qtable(qtable, state, action, next_state, reward)
        total_reward += reward


print(total_reward)
print(qtable)
