
num_mels = 80
num_freq = 1024
sample_rate = 24000
frame_length_ms = 50.
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
hidden_size = 128
embed_size = 256
max_iters = 500
griffin_lim_iters = 60
power = 1.5
outputs_per_step = 3
teacher_forcing_ratio = 1.0

n_layers=1
dropout=0.
decay_step = [500000, 1000000, 2000000]
log_step = 100
save_step = 2000

cleaners='english_cleaners'

data_path = '../Corpora/btec/wav/train/en'
output_path = './result'
checkpoint_path = './model_new'
max_length=100
max_duration=5
att_type="general"
dec_type="Bahdanau"
rnn="LSTM"
enc_type="BASIC"#"CHBG"#
enc_drop=[0.1,0.4]
bidirectional=[True,False]
enc_rnn=["LSTM","GRU"]
#enc_h_size=[256,128]
voc_type="CHBG"#WAVNET
