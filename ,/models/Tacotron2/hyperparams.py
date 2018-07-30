# Text
eos = '~'
pad = '_'
chars = pad + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? ' + eos
unk_idx = len(chars)

# Audio
sample_rate = 22050  # hz
fft_frame_size = 50.0  # ms
fft_hop_size = 12.5  # ms
num_mels = 80  # filters
min_freq = 125  # hz
max_freq = 7600  # hz
floor_freq = 0.01  # reference freq for power to db conversion
spectrogram_pad = 0.0  # change to -80.0? (-80 is the mel value of a window of zeros in the time dim (wave))
max_duration=5
max_length=100
# Encoder
num_chars = len(chars) + 1
padding_idx = 0
embedding_dim = 512

# Decoder
# max_iters = 50

# training
teacher_forcing_ratio = 1.0

# SGDR
cycle_length = 2000
min_lr = 1e-5
max_lr = 1e-3
weight_decay = 1e-6  # l2 reg

# max_grad_norm = 10.0

# WavnetVocoder
embed_size=256
embed_drop=0.
bidirectional=[True,False]
enc_h_size=256
rnn=["LSTM","GRU"]
enc_drop=[0.,0.]

num_mels = 80
num_freq = 1024
sample_rate = 24000
frame_length_ms = 50.
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
max_iters = 200
griffin_lim_iters = 60
power = 1.5
outputs_per_step = 5

max_duration=5
max_length=100
lr=0.001
epochs=50

embedding_size=256
hidden_size=512
n_layers=2
dropout=0.8
#dec_type='Bahdanau'
dec_type='Luong'
att_type='general'
