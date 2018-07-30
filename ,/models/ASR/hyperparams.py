# Audio
num_mels = 80
pre_drop=[0.1,0.1,0.3]
enc_h_size=256
enc_drop=[0.,0.3]
bidirectional=[True,False]

rnn=["LSTM","GRU"]
att_type='general'

embed_size=256
embed_drop=0.3
dec_h_size=256
dec_type='Luong'#'Bahdanau'
dec_drop=[0.3,0.1]
EOS=2
