# Audio
import models.ASR.hyperparams as asr
import models.NMT.hyperparams as nmt
asr_h_size=asr.enc_h_size
nmt_h_size=nmt.enc_h_size

drop=[0.2,0.4]
rnn=["LSTM","LSTM"]
bidirectional=[True,True]
