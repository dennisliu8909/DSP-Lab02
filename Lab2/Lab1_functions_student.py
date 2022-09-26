from operator import le
from turtle import left
from winreg import LoadKey
import numpy as np
import math

def pre_emphasis(signal, coefficient = 0.95):

    return np.append(signal[0], signal[1:] - coefficient*signal[:-1])

def STFT(time_signal, num_frames, num_FFT, frame_step, frame_length, signal_length, verbose=False):
    padding_length = int((num_frames - 1) * frame_step + frame_length)
    padding_zeros = np.zeros((padding_length - signal_length,))
    padded_signal = np.concatenate((time_signal, padding_zeros))

    # split into frames
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames*frame_step, frame_step), (frame_length, 1)).T
    indices = np.array(indices,dtype=np.int32)

    # slice signal into frames
    frames = padded_signal[indices]
    # apply window to the signal
    frames *= np.hamming(frame_length)

    # FFT
    complex_spectrum = np.fft.rfft(frames, num_FFT).T
    print(complex_spectrum.shape)
    absolute_spectrum = np.abs(complex_spectrum)
    
    if verbose:
        print('Signal length :{} samples.'.format(signal_length))
        print('Frame length: {} samples.'.format(frame_length))
        print('Frame step  : {} samples.'.format(frame_step))
        print('Number of frames: {}.'.format(len(frames)))
        print('Shape after FFT: {}.'.format(absolute_spectrum.shape))

    return absolute_spectrum

def mel2hz(mel):
    '''
    Transfer Mel scale to Hz scale
    '''
    ###################
    hz = (10 ** (mel / 2595) - 1 ) * 700
    ###################

    return hz

def hz2mel(hz):
    '''
    Transfer Hz scale to Mel scale
    '''
    ###################
    mel = 2595 * math.log((1 + hz / 700), 10)
    ###################

    return mel

def get_filter_banks(num_filters, num_FFT, sample_rate, freq_min = 0, freq_max = None):
    ''' Mel Bank
    num_filters: filter numbers
    num_FFT: number of FFT quantization values
    sample_rate: as the name suggests
    freq_min: the lowest frequency that mel frequency include
    freq_max: the Highest frequency that mel frequency include
    '''
    # convert from hz scale to mel scale
    low_mel = hz2mel(freq_min)
    high_mel = hz2mel(freq_max)

    # define freq-axis
    mel_freq_axis = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_freq_axis = mel2hz(mel_freq_axis)

    # Mel triangle bank design (Triangular band-pass filter banks)
    bins = np.floor((num_FFT + 1) * hz_freq_axis / sample_rate) # (9, 16, 25, 35, 47, 63, ...)
    fbanks = np.zeros((num_filters, int(num_FFT / 2 + 1)))      # each filter m has triangle wave

    ###################
    # fbanks[m, k]
    for m in range (1, num_filters + 1):
        left_boundry = int(bins[m - 1])
        mid_pt = int(bins[m])
        right_boundry = int(bins[m + 1])
        for k in range (left_boundry, mid_pt):
            fbanks[m - 1, k] = (k - left_boundry) / (mid_pt - left_boundry)
        for k in range (mid_pt, right_boundry):
            fbanks[m - 1, k] = (right_boundry - k) / (right_boundry - mid_pt)
    ###################

    return fbanks