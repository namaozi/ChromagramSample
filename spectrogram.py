#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import scikits.audiolab as al
#⚠ wave読み込みにはscikits.audiolab.wavreadがオススメです。
#私はwaveというパッケージを先に試しましたが,wave.readframesの挙動がおかしかったので使用をやめました。

import functions as fn

"""
スペクトログラムを計算してプロットします
"""
### 楽曲データ読み込み(scikits.audiolab使用)
# data : ここにwavデータがnumpy.ndarrayとして保持されます。
# sampling_rate : 大半のwav音源のサンプリングレートは44.1kHzです
# fmt : フォーマットはだいたいPCMでしょう
file_path = "audios/harmony1.wav"
data, sampling_rate, fmt = al.wavread(file_path)

# ステレオファイルをモノラル化します
x = fn.monauralize(data)

NFFT = 1024 # フレームの大きさ
OVERLAP = NFFT / 2 # 窓をずらした時のフレームの重なり具合. half shiftが一般的らしい
frame_length = data.shape[0] # wavファイルの全フレーム数
time_song = float(frame_length) / sampling_rate  # 波形長さ(秒)
time_unit = 1 / float(sampling_rate) # 1サンプルの長さ(秒)

# 💥 1.
# FFTのフレームの時間を決めていきます
# time_rulerに各フレームの中心時間が入っています
start = (NFFT / 2) * time_unit
stop = time_song
step =  (NFFT - OVERLAP) * time_unit
time_ruler = np.arange(start, stop, step)

# 💥 2.
# 窓関数は周波数解像度が高いハミング窓を用います
window = np.hamming(NFFT)

spec = np.zeros([len(time_ruler), 1 + (NFFT / 2)]) #転置状態で定義初期化
pos = 0

for fft_index in range(len(time_ruler)):
    # 💥 1.フレームの切り出します
    frame = x[pos:pos+NFFT]
    # フレームが信号から切り出せない時はアウトです
    if len(frame) == NFFT:
        # 💥 2.窓関数をかけます
        windowed = window * frame
        # 💥 3.FFTして周波数成分を求めます
        # rfftだと非負の周波数のみが得られます
        fft_result = np.fft.rfft(windowed)
        # 💥 4.周波数には虚数成分を含むので絶対値をabsで求めてから2乗します
        # グラフで見やすくするために対数をとります
        fft_data = np.log(np.abs(fft_result) ** 2)
        # fft_data = np.log(np.abs(fft_result))
        # fft_data = np.abs(fft_result) ** 2
        # fft_data = np.abs(fft_result)
        # これで求められました。あとはspecに格納するだけです
        for i in range(len(spec[fft_index])):
            spec[fft_index][-i-1] = fft_data[i]

        # 💥 4. 窓をずらして次のフレームへ
        pos += (NFFT - OVERLAP)

### プロットします
# matplotlib.imshowではextentを指定して軸を決められます。aspect="auto"で適切なサイズ比になります
plt.imshow(spec.T, extent=[0, time_song, 0, sampling_rate/2], aspect="auto")
plt.xlabel("time[s]")
plt.ylabel("frequency[Hz]")
plt.colorbar()
plt.show()
