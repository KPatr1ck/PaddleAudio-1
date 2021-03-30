import paddle
import numpy as np
import librosa
__all__ = ['mel_spect','linear_spect','log_spect']

def mel_spect(y,sample_rate=16000,
                window_size=512,
                hop_length=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                window = 'hann',
                center = True,
                pad_mode = 'reflect',
                ref = 1.0,
                amin = 1e-10,
                top_db = None):
    

    s = librosa.stft(y,n_fft=window_size,
                               hop_length=hop_length,
                               win_length=window_size,
                               window=window,
                               center=center, pad_mode=pad_mode)

    power = np.abs(s)**2
    melW = librosa.filters.mel(sr=sample_rate,
                               n_fft=window_size,
                               n_mels=mel_bins,
                fmin=fmin, fmax=fmax)
    mel = np.matmul(melW,power)
    db = librosa.power_to_db(mel,ref=ref,amin=amin,top_db=None)
    #db = db.transpose()
    return db


def linear_spect(y,sample_rate=16000,
                window_size=512,
                hop_length=320,
                window = 'hann',
                center = True,
                pad_mode = 'reflect'):
    

    s = librosa.stft(y,n_fft=window_size,
                               hop_length=hop_size,
                               win_length=window_size,
                               window=window,
                               center=center, pad_mode=pad_mode)

    return np.abs(s)[:-1,:] # remove 
def log_spect(y,sample_rate=16000,
                window_size=512,
                hop_length=320,
                window = 'hann',
                center = True,
                pad_mode = 'reflect'):
    

    s = librosa.stft(y,n_fft=window_size,
                               hop_length=hop_size,
                               win_length=window_size,
                               window=window,
                               center=center, pad_mode=pad_mode)
    s = np.abs(s)[:-1,:]
    

    return np.log(1+s)  # remove 
    
    
