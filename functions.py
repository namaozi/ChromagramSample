#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

def monauralize(data):
    #モノラル化
    try:
        if data.shape[1] == 2:
            res = 0.5 * (data.T[0] + data.T[1])
    except:
        res = data
    return res
###

def cos_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
###

def librosa_chroma(file_path="audios/harmony1.wav", sr=44100):
    #インポート(インストールしないと使えません)
    import librosa

    # 読み込み(sr:サンプリングレート)
    y, sr = librosa.load(file_path, sr=sr)

    # 楽音成分とパーカッシブ成分に分けます
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # クロマグラムを計算します
    C = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)

    # プロットします
    plt.figure(figsize=(12,4))
    librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
    plt.title('Chromagram')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    return C
###
