import sys,os
import librosa
import random
import numpy as np
import collections
from scipy import signal
import torch

_mel_basis = None

def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    librosa.output.write_wav(path, wav.astype(np.int16), hp.sample_rate)
    #import pdb; pdb.set_trace()
    #wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #librosa.output.write_wav(path, wav.astype(np.int16), hp.sample_rate)


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
    n_fft = (hp.num_freq - 1) * 2
    return librosa.filters.mel(hp.sample_rate, n_fft, n_mels=hp.num_mels)

def _normalize(S):
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)

def _denormalize(S):
    return (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db

def _stft_parameters():
    n_fft = (hp.num_freq - 1) * 2
    hop_length = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    win_length = int(hp.frame_length_ms / 1000 * hp.sample_rate)
    return n_fft, hop_length, win_length

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def preemphasis(x):

    return signal.lfilter([1, -hp.preemphasis], [1], x)


def inv_preemphasis(x):
    return signal.lfilter([1], [1, -hp.preemphasis], x)


def spectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    return _normalize(S)


def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''

    S = _denormalize(spectrogram)
    S = _db_to_amp(S + hp.ref_level_db)  # Convert back to linear

    return inv_preemphasis(_griffin_lim(S ** hp.power))          # Reconstruct phase

def _griffin_lim(S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(hp.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y

def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def melspectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D)))
    return _normalize(S)

def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(hp.sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)

def _pad_data(x, length):
    _pad = 0
    _pad_size=length-x.shape[0]
    if x.ndim==2:
        ndim=x.shape[1]
        x=x.reshape(x.shape[0]*x.shape[1])
        return np.pad(x, (0, _pad_size*ndim), mode='constant', constant_values=_pad).reshape(length,ndim)
    return np.pad(x, (0,_pad_size), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

def _pad_per_step(inputs):

    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, hp.outputs_per_step - (timesteps % hp.outputs_per_step)]], mode='constant', constant_values=0.0)

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params
