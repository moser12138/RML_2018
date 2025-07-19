import numpy as np
import matplotlib.pyplot as plt

# 生成或加载IQ信号，假设为 (1024, 2) 的数组
iq_signal = np.random.randn(1024, 2)  # 示例随机信号
i_signal = iq_signal[:, 0]
q_signal = iq_signal[:, 1]

# 将 I 和 Q 分量合成为复数信号
complex_signal = i_signal + 1j * q_signal

# 计算信号的FFT
fft_result = np.fft.fft(complex_signal)
fft_freq = np.fft.fftfreq(len(complex_signal))

# 绘制FFT频谱
plt.figure(figsize=(10, 6))
plt.plot(fft_freq, np.abs(fft_result))
plt.title('FFT Spectrum of IQ Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()


from scipy.signal import spectrogram

# 计算信号的短时傅里叶变换（STFT）
frequencies, times, Sxx = spectrogram(complex_signal, fs=1.0, nperseg=256, noverlap=128, nfft=512)

# 绘制STFT时频图
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(np.abs(Sxx)), shading='gouraud')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('STFT Time-Frequency Spectrogram of IQ Signal')
plt.show()

from tftb.processing import WignerVilleDistribution

# 计算Wigner-Ville分布（WVD）
wvd = WignerVilleDistribution(complex_signal)
wvd.run()

# 绘制Wigner-Ville时频图
plt.figure(figsize=(10, 6))
wvd.plot(kind='contour', show_tf=True, cmap='jet')
plt.title('Wigner-Ville Distribution of IQ Signal')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

