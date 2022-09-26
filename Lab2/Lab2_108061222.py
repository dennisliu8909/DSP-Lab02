'''
@Modified by Paul Cho; 10th, Nov, 2020

For NTHU DSP Lab 2022 Autumn
'''

'''PART I Copy paste to the top import section:'''
from msilib.schema import Feature
from Lab2_stft2audio_student import griffinlim
from scipy.fftpack import idct
from scipy.linalg import pinv
'''PART I ENDS HERE.'''
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from librosa.filters import mel as librosa_mel_fn
from scipy.fftpack import dct

from Lab1_functions_student import pre_emphasis, STFT, mel2hz, hz2mel, get_filter_banks

filename = './audio.wav'
source_signal, sr = sf.read(filename) #sr:sampling rate
print('Sampling rate={} Hz.'.format(sr))

### hyper parameters
frame_length = 512                    # Frame length(samples)
frame_step = 256                      # Step length(samples)
emphasis_coeff = 0.95                 # pre-emphasis para
num_bands = [12, 64]                     # Filter number = band number
num_FFT = frame_length                # FFT freq-quantization
freq_min = 0
freq_max = int(0.5 * sr)
signal_length = len(source_signal)    # Signal length

# number of frames it takes to cover the entirety of the signal
num_frames = 1 + int(np.ceil((1.0 * signal_length - frame_length) / frame_step))

##########################
'''
Part I:
(1) Perform STFT on the source signal to obtain one spectrogram (with the provided STFT() function)
(2) Pre-emphasize the source signal with pre_emphasis()
(3) Perform STFT on the pre-emphasized signal to obtain the second spectrogram
(4) Plot the two spectrograms together to observe the effect of pre-emphasis

hint for plotting:
you can use "plt.subplots()" to plot multiple figures in one.
you can use "axis.pcolor" of matplotlib in visualizing a spectrogram. 
'''
#YOUR CODE STARTS HERE:
ori_spectrum = STFT(source_signal, num_frames, num_FFT, frame_step, frame_length, signal_length, False)
signal_pre_emphasis = pre_emphasis(source_signal, 0.95)
pre_ori_spectrum = STFT(signal_pre_emphasis, num_frames, num_FFT, frame_step, frame_length, signal_length, False)

#YOUR CODE ENDS HERE;
##########################

'''
Head to the import source 'Lab1_functions_student.py' to complete these functions:
mel2hz(), hz2mel(), get_filter_banks()
'''

fig1, ax = plt.subplots(1, 2)
fig2, bx = plt.subplots(1, 2)
for i in range(0, 2):
    # get Mel-scaled filter
    fbanks = get_filter_banks(num_bands[i], num_FFT , sr, freq_min, freq_max)
    xaxis = np.arange(0, freq_max, freq_max / (frame_step + 1))

    ##########################
    '''
    Part II:
    (1) Convolve the pre-emphasized signal with the filter
    (2) Convert magnitude to logarithmic scale
    (3) Perform Discrete Cosine Transform (dct) as a process of information compression to obtain MFCC
        (already implemented for you, just notice this step is here and skip to the next step)
    (4) Plot the filter banks alongside the MFCC
    '''
    #YOUR CODE STARTS HERE:
    features = np.dot(fbanks, pre_ori_spectrum)
    features = np.where(features == 0, np.finfo(float).eps, features)
    features = np.log(features)
    print('spectrum after fbanks', features.shape)

    # step(3): Discrete Cosine Transform
    MFCC = dct(features.T, norm = 'ortho')[:,:num_bands[i]]
    # equivalent to Matlab dct(x)
    # The numpy array [:,:] stands for everything from the beginning to end.
    ax[i].set_title('MFCC of {} banks'.format(num_bands[i]))
    ax[i].set_xlabel('Cepstral coefficient')
    ax[i].set_ylabel('Magnitude')
    ax[i].pcolor(MFCC.T)
    MFCC_random = MFCC[5]
    plt.suptitle('MFCC of random frame')
    bx[i].set_xlabel('Cepstral coefficient')
    bx[i].set_ylabel('Magnitude')
    bx[i].plot(MFCC_random)
# plt.show()
    


#YOUR CODE ENDS HERE;
##########################

'''PART III Ciouopy paste to the very bottom after all your prevs code (where you have the MFCC obtained):'''
'''
(1) Perform inverse DCT on MFCC (already done for you)
(2) Restore magnitude from logarithmic scale (i.e. use exponential)
(3) Invert the fbanks convolution
(4) Synthesize time-domain audio with Griffin-Lim
(5) Get STFT spectrogram of the reconstructed signal and compare it side by side with the original signal's STFT spectrogram
    (please convert magnitudes to logarithmic scale to better present the changes)
'''

# inverse DCT (done for you)
inv_DCT = idct(MFCC, norm = 'ortho')
print('Shape after iDCT:', inv_DCT.shape)

# mag scale restoration:
###################
IDCT_feature = inv_DCT.T
IDCT_feature = np.exp(IDCT_feature)
###################

# inverse convoluation against fbanks (mind the shapes of your matrices):
fbanks_AAT = np.dot(fbanks, fbanks.T)
print(fbanks_AAT.shape)
fbanks_AATinv = np.linalg.pinv(fbanks_AAT)
inv_spectrogram = np.dot(fbanks.T, fbanks_AATinv)
inv_spectrogram = np.dot(inv_spectrogram, IDCT_feature)

print('Shape after inverse convolution:', inv_spectrogram.shape)


# signal restoration to time domain (You only have to finish griffinlim() in 'stft2audio_student.py'):
inv_audio = griffinlim(inv_spectrogram, n_iter=32, hop_length=frame_step, win_length=frame_length)
sf.write('./reconstructed.wav', inv_audio, samplerate=int(sr*512/frame_length))
reconstructed_spectrum = STFT(inv_audio, num_frames, num_FFT, frame_step, frame_length, len(inv_audio), verbose=False)

# scale and plot and compare original and reconstructed signals
# scale (done for you):
absolute_spectrum = np.where(ori_spectrum == 0, np.finfo(float).eps, ori_spectrum)
absolute_spectrum = np.log(absolute_spectrum)
reconstructed_spectrum = np.where(reconstructed_spectrum == 0, np.finfo(float).eps, reconstructed_spectrum)
reconstructed_spectrum = np.log(reconstructed_spectrum)

#plot:
###################
subfig, ax = plt.subplots(1, 2)
plt.suptitle('Original signal v.s. Reconstructed signal')
ax[0].set_xlabel('frame')
ax[0].set_ylabel('frequency band')
ax[0].pcolor(absolute_spectrum)
ax[1].set_xlabel('frame')
ax[1].set_ylabel('frequency band')
ax[1].pcolor(reconstructed_spectrum)
plt.show()
###################

'''PART III ENDS  HERE.'''

