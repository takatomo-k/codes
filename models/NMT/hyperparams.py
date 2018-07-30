# Audio
embed_size=256
embed_drop=0.3
bidirectional=[True,False]
enc_h_size=512
rnn=["LSTM","GRU"]
enc_drop=[0.1,0.]

att_type='general'

dec_h_size=512
dec_type='Luong'#'Bahdanau'
dec_drop=[0.1,0.6]
EOS=2
