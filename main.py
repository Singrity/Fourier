import sys

import numpy as np
import matplotlib.pyplot as plt
import wave


def i_fft(x):
    _x = np.asarray(x, dtype=float)
    N = _x.shape[0]
    return 1/N * np.conjugate(fft(np.conjugate(x)))


def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N % 2 > 0:
        raise ValueError("Must be a power of 2")
    elif N <= 2:
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)

        return np.concatenate([X_even + terms[:N // 2] * X_odd,
                               X_even + terms[N // 2:] * X_odd])


def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

# def idft(x):
#     x = np.asarray(x, dtype=float)
#     N = x.shape[0]
#     n = np.arange(N)
#     k = n.reshape((N, 1))
#     M = np.exp(2j * np.pi * k * n / N)
#     return np.dot(1/N * M, x)


def get_params_from_signal(file):
    obj = wave.open(file, 'rb')
    Fs = obj.getframerate()
    N = obj.getnframes()

    window_size = int(input(f"Print size of window (only power of 2, size of signal = {N})"))

    if window_size > N:
        raise ValueError(f"size: {window_size} > N: {N}")
    t_step = 1 / Fs  # time interval
    f_step = Fs / window_size  # freq interval
    signal = obj.readframes(window_size)
    signal = np.frombuffer(signal, "int16")

    t = np.linspace(0, window_size * t_step, window_size)  # time steps
    f = np.linspace(0, window_size * f_step, window_size)  # freq steps
    if obj.getnchannels() == 2:
        print("Stereo signal was detected!!!")
        sys.exit(0)
    return signal, N, t, f, window_size


def sub_plot(size):
    f_plot = f[0:int(size / 2 + 1)]
    x_mag_plot = 2 * X_numpy_mag[0:int(size / 2 + 1)]
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1)
    fig.suptitle("Numpy")

    ax1.plot(t, signal)
    ax2.plot(f_plot, x_mag_plot, '.-')

    ax3.plot(t, X_i_numpy.real)

    ax1.set_title("Original")
    ax2.set_title("FFT")
    ax3.set_title("FFT -> Original")

    ax1.set_xlabel("time (s)")
    ax2.set_xlabel("freq (Hz)")
    ax3.set_xlabel("time (s)")
    # ax2.set_xlim(0, 20000)

    ax1.grid()
    ax2.grid()
    ax3.grid()

    plt.tight_layout()


def plot(size):
    f_plot = f[0:int(size / 2 + 1)]
    x_mag_plot = 2 * X_mag[0:int(size / 2 + 1)]

    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1)

    fig.suptitle("Own")

    ax1.plot(t, signal)
    ax2.plot(f_plot, x_mag_plot, '.-')

    ax3.plot(t, X_i.real)

    ax1.set_title("Original")
    ax2.set_title("FFT")
    ax3.set_title("FFT -> Original")

    ax1.set_xlabel("time (s)")
    ax2.set_xlabel("freq (Hz)")
    ax3.set_xlabel("time (s)")
    # ax2.set_xlim(0, 20000)

    ax1.grid()
    ax2.grid()
    ax3.grid()

    plt.tight_layout()

    sub_plot(size)

    plt.show()


if __name__ == '__main__':
    signal, N, t, f, window_size = get_params_from_signal("samples/с4.wav")

    # using numpy
    X_numpy = np.fft.fft(signal)
    X_i_numpy = np.fft.ifft(X_numpy)
    X_numpy_mag = np.abs(X_numpy) / N

    # own
    print(len(signal))
    X = fft(signal)

    # X = dft(signal)
    X_i = i_fft(X)
    X_mag = np.abs(X) / N

    # Plot
    plot(window_size)

