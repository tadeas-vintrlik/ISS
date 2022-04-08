import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy.signal import spectrogram, butter, lfilter
from scipy.io import wavfile
from functools import reduce

# Load the signal, get basic information
fs, x = wavfile.read('../audio/xvintr04.wav')
t = np.arange(x.size) / fs
print("Number of samples:", x.size)
print("Duration in seconds:", x.size / fs)
print("Minimum value:", min(x))
print("Maximum value:", max(x))

# Plot the entire signal
plt.figure()
plt.plot(t, x)
plt.ylabel("x(t)")
plt.xlabel("t [s]")
plt.title("Načtený signál")
plt.savefig("./plots/01.png")

# Normalize the signal
x = x - np.mean(x)
x = x / max(np.abs(x))
assert(max(x) <= 1)
assert(min(x) >= -1)

# Create a matrix of overlaping frames
f = np.array([x[i:i+1024] for i in range(0, x.size - 512, 512)])
ft = np.array([t[i:i+1024] for i in range(0, x.size - 512, 512)])
assert(len(f) == len(ft))
assert(len(f) == 85)
assert(len(f[0]) == 1024)

fi = 39

# Plot one of the middle frames
plt.figure()
plt.plot(ft[fi], f[fi])
plt.ylabel("x(t)")
plt.xlabel("t [s]")
plt.title("40. Rámec")
plt.savefig("./plots/02.png")

##
# My implementation of Discrete Fourier Transform using matrix multiplication
# Only works on discrete signals of length 1024
# Very slow compared to np.fft
def my_dft(x):
    # Prepare the base matrix
    n = np.arange(1024)
    k = np.arange(1024).reshape((1024, 1))
    b = n * k
    b = np.exp(-1j * 2 * np.pi * b / 1024)

    # Multiply the base matrix with the signal vector
    return np.dot(b, x)

# Plot the DFT results compare mine and np.fft.fft
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(np.arange(512) * fs, np.abs(my_dft(f[fi])[0:512]))
plt.title("Moje DFT")
plt.xlabel("f [Hz]")
plt.subplot(1, 2, 2)
plt.plot(np.arange(512) * fs, np.abs(np.fft.fft(f[fi])[0:512]))
plt.title("NumPy FFT")
plt.xlabel("f [Hz]")
plt.savefig("./plots/03.png")

# Create the spectrogram
freqs, time, vals = spectrogram(x, fs)
vals = 10 * np.log10(vals) 

# Plot the spectrogram
plt.figure()
plt.pcolormesh(time, freqs, vals)
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
cbar = plt.colorbar()
plt.title("Spektrogram původního signálu")
cbar.set_label('Logaritmický výkon [dB]')
plt.savefig("./plots/04.png")

# From the spectrogram read the frequencies of the cosines
# 2550Hz, 1925Hz, 1300Hz and 650Hz
cos_freqs = [2550, 1925, 1300, 650]
cos4 = reduce(operator.add, [np.cos(2*np.pi*f*t) for f in cos_freqs], 0)/len(cos_freqs)

freqs, time, vals = spectrogram(cos4, fs)
vals = 10 * np.log10(vals) 
plt.figure()
plt.pcolormesh(time, freqs, vals)
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
cbar = plt.colorbar()
plt.title("Spektrogram rušivých cosinů")
cbar.set_label('Logaritmický výkon [dB]')
plt.savefig("./plots/05.png")

def savewav(path, fs, signal):
    # Since the signal is normalized to <-1,1> increate to max int16 size
    wavfile.write(path, fs, (signal * 32767).astype(np.int16))

savewav("../audio/4cos.wav", fs, cos4)

## Used the following example from scipy API docs
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
def bandstop_filter(signal, freq, fs):
    # Get the a and b coefficients of the filter
    nyq = 0.5 * fs # Nyquist frequency used by scipy butter
    width = 75
    low = (freq-width) / nyq
    high = (freq+width) / nyq
    b, a = butter(5, [low, high], btype='bandstop', output='ba')
    return lfilter(b, a, signal)

clean = x
for f in cos_freqs:
    clean = bandstop_filter(clean, f, fs)

savewav("../audio/clean_bandstop.wav", fs, clean)
