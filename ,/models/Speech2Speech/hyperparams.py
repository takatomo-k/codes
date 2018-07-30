# Audio
import new_models.Asr.hyperparams as asr
import new_models.Nmt.hyperparams as nmt
tts_h_size=256
nmt_h_size=nmt.enc_h_size

drop=[0.2,0.2]
rnn=["LSTM","LSTM"]
bidirectional=[False,False]
