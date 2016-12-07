#coding:utf-8
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import librosa #インストールしてください!

import functions as fn

file_path = "audios/"
file_name = "harmony1.wav"

# クロマグラムを求めます
chroma = fn.librosa_chroma(file_path + file_name)

TONES = 12 # ピッチクラス,音の種類の数
sampling_rate = 44100 #音源依存

# "この設定では",こんな感じで時間軸設定を求められます
# (詳しくはドキュメントを読んで下さい)
time_unit = 512.0 / 44100 # 1フレームのクロマグラムの長さ
# stop = time_unit * (chroma.shape[1] - 1)
stop = time_unit * (chroma.shape[1])
time_ruler = np.arange(0, stop, time_unit)

###コードのテンプレートベクトルです
#メジャーとマイナーだけを考えます
#脳筋コード,時間が無かったので…
#順番を保ちたいのでOrderdDictを使います
one_third = 1.0/3
chord_dic = OrderedDict()
chord_dic["C"] = [one_third, 0,0,0, one_third, 0,0, one_third, 0,0,0,0]
chord_dic["Db"] = [0, one_third, 0,0,0, one_third, 0,0, one_third, 0,0,0]
chord_dic["D"] = [0,0, one_third, 0,0,0, one_third, 0,0, one_third, 0,0]
chord_dic["Eb"] = [0,0,0, one_third, 0,0,0, one_third, 0,0, one_third, 0]
chord_dic["E"] = [0,0,0,0, one_third, 0,0,0, one_third, 0,0, one_third]
chord_dic["F"] = [one_third, 0,0,0,0, one_third, 0,0,0, one_third, 0,0]
chord_dic["Gb"] = [0, one_third, 0,0,0,0, one_third, 0,0,0, one_third, 0]
chord_dic["G"] = [0,0, one_third, 0,0,0,0, one_third, 0,0,0, one_third]
chord_dic["Ab"] = [one_third, 0,0, one_third, 0,0,0,0, one_third, 0,0,0]
chord_dic["A"] = [0, one_third, 0,0, one_third, 0,0,0,0, one_third, 0,0]
chord_dic["Bb"] = [0,0, one_third, 0,0, one_third, 0,0,0,0, one_third, 0]
chord_dic["B"] = [0,0,0, one_third, 0,0, one_third, 0,0,0,0, one_third]
chord_dic["Cm"] = [one_third, 0,0, one_third, 0,0,0, one_third, 0,0,0,0]
chord_dic["Dbm"] = [0, one_third, 0,0, one_third, 0,0,0, one_third, 0,0,0]
chord_dic["Dm"] = [0,0, one_third, 0,0, one_third, 0,0,0, one_third, 0,0]
chord_dic["Ebm"] = [0,0,0, one_third, 0,0, one_third, 0,0,0, one_third, 0]
chord_dic["Em"] = [0,0,0,0, one_third, 0,0, one_third, 0,0,0, one_third]
chord_dic["Fm"] = [one_third, 0,0,0,0, one_third, 0,0, one_third, 0,0,0]
chord_dic["Gbm"] = [0, one_third, 0,0,0,0, one_third, 0,0, one_third, 0,0]
chord_dic["Gm"] = [0,0, one_third, 0,0,0,0, one_third, 0,0, one_third, 0]
chord_dic["Abm"] = [0,0,0, one_third, 0,0,0,0, one_third, 0,0, one_third]
chord_dic["Am"] = [one_third, 0,0,0, one_third, 0,0,0,0, one_third, 0,0]
chord_dic["Bbm"] = [0, one_third, 0,0,0, one_third, 0,0,0,0, one_third, 0]
chord_dic["Bm"] = [0,0, one_third, 0,0,0, one_third, 0,0,0,0, one_third]

prev_chord = 0
sum_chroma = np.zeros(TONES)
estimate_chords = []

result = np.zeros((TONES * 2, 8))

for time_index, time in enumerate(time_ruler):
    # 今は何番目のコードを解析しているのか
    # 2秒おきに変わるので2で割って求めます
    nth_chord = int(time) / 2

    # 次の2秒間に移る時に,前の2秒間のコードを推定します
    if nth_chord != prev_chord:
        maximum = -100000
        this_chord = ""
        # コサイン類似度が最大になるコードを調べます
        for chord_index, (name, vector) in enumerate(chord_dic.iteritems()):
            similarity = fn.cos_sim(sum_chroma, vector)
            result[chord_index][nth_chord - 1] = similarity
            if similarity > maximum:
                maximum = similarity
                this_chord = name
        # 初期化、推定したコードを格納します
        sum_chroma = np.zeros(TONES)
        estimate_chords.append(this_chord)

    else:
        # chromaのshapeに注意しながら足していきます
        for i in range(TONES):
            sum_chroma[i] += chroma[i][time_index]

    # 更新
    prev_chord = nth_chord
###

# 最終結果です
print estimate_chords

###がんばってプロットします
axis_x = np.arange(0, 16, 2)
bar_width = 0.07
colors = ["#ff9999", "#ffaf95","#fabb92","#ffd698","#fae991","#c1fc97","#97fac8","#96f9f5","#98e1fb","#9cb2ff","#b79bfe","#fa96f9", "#b36a6a", "#ab7361","#aa7d61","#ad9165","#b4a765","#8ab66b","#6ab48f","#68b0ad","#689fb3","#6979b0","#7462a3","#aa62a9"]
for i, (name, vector) in enumerate(chord_dic.iteritems()):
    plt.bar(axis_x + bar_width * i, result[i], color=colors[i], width = bar_width, label = name, align = "center")

plt.legend()
plt.xticks(axis_x + bar_width / 24)
plt.show()
