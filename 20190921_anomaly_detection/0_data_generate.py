# -*- coding: utf-8 -*-
"""
異常検知用 データ生成
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_next_pv(sv, pv, k1, k2):
    """
    設定値svから振れを含んだ実際の値pvを求める
    sv: 設定値
    pv: 1ステップ前の実際値
    k1: svとpvの差異上限を決める指数 (-)
    k2: pvの変動距離を決める指数 (-)
    """
    ## pvの変動距離計算
    ## 変動距離はsvとpvの差異上限値k1*svにk2をかけたものを3×標準偏差とした正規分布から求める
    stepsize_3s = k2 * k1 * sv # k1*svがsvとpvの差異上限値。これにk2をかけたものを3σ
    stepsize = abs(np.random.randn()*stepsize_3s/3.0) # 3σを用い正規分布から変動距離決定

    ## +に変動するか、-に変動するかを確率的に決定
    ## 変動確率はsvとpvの差異の関数とする
    prov =  -0.5/(sv*k1)*abs(pv-sv) + 0.5
    if prov < 0: prov = 0 # 差異上限をこえたら、それ以上差異が広がらないようにする

    if pv-sv >= 0: # pvがsvより大きいか小さいかで正負の確率を切り替える
        coef = np.random.choice([-1, 1], p=[1-prov, prov])
    else:
        coef = np.random.choice([-1, 1], p=[prov, 1-prov])
        
    return pv + coef * stepsize

def calc_to(f, tf, p, tm):
    """
    原料流量、原料温度、反応圧力、熱媒体温度から出口温度を求める
    """
    return tm + 0.5*(1000-f) + (tf-50) + 25*(12.0-p)

def calc_to_with_anomaly(to):
    """
    正常時の出口温度から異常を含んだ出口温度を求める
    あるステップ(運転日数)から異常温度上昇が追加される
    """
    step = 700 # 異常兆候開始ステップ
    anomaly_effect = [0 if i<step else 0.01 * (i-step)**1.4 for i in np.arange(len(to))]
    return to + anomaly_effect

## 各制御値の設定値 1要素が1日分のデータ
f_sv  = [1000]*100+[1010]*10+[1020]*10+[1030]*10+[1040]*20+[1050]*100 +\
        [1040]*10+[1030]*10+[1020]*10+[1010]*10+[1000]*210+[1010]*10+[1020]*10 +\
        [1030]*10+[1040]*50+[1030]*10+[1020]*10+[1010]*400
tf_sv = [50]*1000
p_sv  = [12.0]*1000
tm_sv = [300]*100+[302]*10+[304]*10+[306]*10+[308]*20+[306]*100+[304]*250+[306]*20 +\
        [308]*50+[306]*30+[304]*400

## 各制御値の実際の値。プラントでは設定値ちょうどにならず多少上下に振れることを考慮
# 初期値
f_pv  = [1000]
tf_pv = [50]
p_pv  = [12.0]
tm_pv = [300]

# 設定値から振れを含んだ実際の値を求める
for fs, tfs, ps, tms in zip(f_sv, tf_sv, p_sv, tm_sv):
    fp  = f_pv[-1]
    tfp = tf_pv[-1]
    pp  = p_pv[-1]
    tmp = tm_pv[-1]
    f_pv.append(generate_next_pv(fs, fp, 0.02, 1))
    tf_pv.append(generate_next_pv(tfs, tfp, 0.2, 0.2))
    p_pv.append(generate_next_pv(ps, pp, 0.05, 0.1))
    tm_pv.append(generate_next_pv(tms, tmp, 0.02, 0.2))

## データを外部ファイルに保存するためにnumpy.arrayに変換
f_sv  = np.array(f_sv)
tf_sv = np.array(tf_sv)
p_sv  = np.array(p_sv)
tm_sv = np.array(tm_sv)
f_pv  = np.array(f_pv)
tf_pv = np.array(tf_pv)
p_pv  = np.array(p_pv)
tm_pv = np.array(tm_pv)
to_pv = calc_to(f_pv, tf_pv, p_pv, tm_pv)
to_an = calc_to_with_anomaly(to_pv)

## ファイル保存
np.save('f_sv', f_sv)
np.save('tf_sv', tf_sv)
np.save('p_sv', p_sv)
np.save('tm_sv', tm_sv)
np.save('f_pv', f_pv)
np.save('tf_pv', tf_pv)
np.save('p_pv',  p_pv)
np.save('tm_pv', tm_pv)
np.save('to_pv', to_pv)
np.save('to_an', to_an)

""" 
## プロット用。使用する際はコメントアウトを外す
fig = plt.figure(figsize=(16,10))
ax1 = fig.add_subplot(421)
ax2 = fig.add_subplot(423)
ax3 = fig.add_subplot(425)
ax4 = fig.add_subplot(427)
ax5 = fig.add_subplot(422)
ax1.set_ylabel("Flow rate, kg/h")
ax2.set_ylabel("Feed temp., ℃")
ax3.set_ylabel("Pressure, MPa")
ax4.set_ylabel("Medium temp., ℃")
ax4.set_xlabel("Step")
ax1.set_ylim(950, 1100)
ax2.set_ylim(40, 60)
ax3.set_ylim(11.5, 12.5)
ax4.set_ylim(290, 315)
ax5.set_ylim(260, 340)
ax5.grid(axis='y')
ax1.plot(f_pv)
ax1.plot(f_sv)
ax2.plot(tf_pv)
ax2.plot(tf_sv)
ax3.plot(p_pv)
ax3.plot(p_sv)
ax4.plot(tm_pv)
ax4.plot(tm_sv)
ax5.plot(to_pv)
ax5.plot(to_an)
plt.show()
"""
