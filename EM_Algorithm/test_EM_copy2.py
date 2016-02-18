# -*- coding: utf-8 -*-

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#平均m, 分散vのガウス分布
def gaussian(x, m, v):
    p = math.exp(- pow(x - m, 2) / (2 * v)) / math.sqrt(2 * math.pi * v)
    return p

#Eステップ
def e_step(xs, ms, vs, p):
    burden_rates = []
    for x in xs:
        d = (1 - p) * gaussian(x, ms[0], vs[0]) + p * gaussian(x, ms[1], vs[1])
        n = p * gaussian(x, ms[1], vs[1])
        burden_rate = n / d
        burden_rates.append(burden_rate)
    return burden_rates


#Mステップ
def m_step(xs, burden_rates):
    d = sum([1 - r for r in burden_rates])
    n = sum([(1 - r) * x for x, r in zip(xs, burden_rates)])
    mu1 = n / d

    n = sum([(1 - r) * pow(x - mu1, 2) for x, r in zip(xs, burden_rates)])
    var1 = n / d

    d = sum(burden_rates)
    n = sum([r * x for x, r in zip(xs, burden_rates)])
    mu2 = n / d

    n = sum(r * pow(x - mu2, 2) for x, r in zip(xs, burden_rates))
    var2 = n / d

    N = len(xs)
    p = sum(burden_rates) / N

    return [mu1, mu2], [var1, var2], p


#対数尤度関数
def calc_log_likelihood(xs, ms, vs, p):
    s = 0
    for x in xs:
        g1 = gaussian(x, ms[0], vs[0])
        g2 = gaussian(x, ms[1], vs[1])
        s += math.log((1 - p) * g1 + p * g2)
    return s

#ヒストグラムをプロットする関数
def draw_hist(xs, bins):
    plt.hist(xs, bins=bins, normed=True, alpha=0.5)

def main():
    #データセットの1列目を読み込む
    fp = open("faithful.txt")
    data = []
    for row in fp:
        data.append(float((row.split()[0])))
    fp.close()
    #mu, vs, pの初期値を設定する    p = 0.5
    ms = [random.choice(data), random.choice(data)]
    vs = [np.var(data), np.var(data)]
    T = 50  #反復回数
    ls = []  #対数尤度関数の計算結果を保存しておく
    #EMアルゴリズム
    for t in range(T):
        burden_rates = e_step(data, ms, vs, p)
        ms, vs, p = m_step(data, burden_rates)
        ls.append(calc_log_likelihood(data, ms, vs, p))

    print("predict: mu1={0}, mu2={1}, v1={2}, v2={3}, p={4}".format(
        ms[0], ms[1], vs[0], vs[1], p))
    #結果のプロット
    plt.subplot(211)
    xs = np.linspace(min(data), max(data), 200)
    norm1 = mlab.normpdf(xs, ms[0], math.sqrt(vs[0]))
    norm2 = mlab.normpdf(xs, ms[1], math.sqrt(vs[1]))
    draw_hist(data, 20)
    plt.plot(xs, (1 - p) * norm1 + p * norm2, color="red", lw=3)
    plt.xlim(min(data), max(data))
    plt.xlabel("x")
    plt.ylabel("Probability")

    plt.subplot(212)
    plt.plot(np.arange(len(ls)), ls)
    plt.xlabel("step")
    plt.ylabel("log_likelihood")
    plt.show()

if __name__ == '__main__':
    main()
