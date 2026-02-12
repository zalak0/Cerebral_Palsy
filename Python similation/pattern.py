import numpy as np
import librosa
import matplotlib.pyplot as plt

def plot_waveform(wav_path, sr=16000, show_rms=True, hop=256, n_fft=1024, trim=False):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    if trim:
        y, _ = librosa.effects.trim(y, top_db=25)

    t = np.arange(len(y)) / sr

    plt.figure()
    plt.plot(t, y)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform: {wav_path}")

    if show_rms:
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
        tr = np.arange(len(rms)) * (hop / sr)

        # 为了能叠在同一张图上，把 RMS 归一化到波形幅度范围
        y_max = np.max(np.abs(y)) + 1e-9
        rms_scaled = (rms / (np.max(rms) + 1e-9)) * y_max

        plt.plot(tr, rms_scaled)  # RMS 包络（缩放后叠加）
        # 注：不指定颜色，matplotlib 会自动选色

    plt.tight_layout()
    plt.show()


# ===== 你在这里填文件 =====
test_files = [
    #r"C:\Users\Stacee\Desktop\cry\cry4.wav",
    r"C:\Users\Stacee\Desktop\voice\laugh2.wav",
    #r"C:\Users\Stacee\Desktop\talk\talk11.wav",
    #r"C:\Users\Stacee\Desktop\cry\hungry\0D1AD73E-4C5E-45F3-85C4-9A3CB71E8856-1430742197-1.0-m-04-hu.wav",
    #r"C:\Users\Stacee\Desktop\laugh\laugh_1.m4a_2.wav"
]


for f in test_files:
    plot_waveform(f, sr=16000, show_rms=True, trim=False)
