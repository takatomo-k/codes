#-*- coding: utf-8 -*-
import random,sys,os,torch
from torch import nn
from modules.Sequential_modules import *
from modules.Basic_modules import *
from modules.Attention_modules import *
from models.Tacotron.network import Tacotron
from models.NMT.network import NMT
import models.MTTTS.hyperparams as hp
from torch.nn import functional as F
class MTTTS(nn.Module):
    def __init__(self,src_vocab_size,trg_vocab_size):
        super(MTTTS, self).__init__()
        self.nmt=NMT(src_vocab_size,trg_vocab_size)
        self.tts=Tacotron(trg_vocab_size)

    def __call__(self,src_txt=None,trg_txt=None,trg_mel=None,teacher_forcing_ratio=1.,mode="normal"):
        nmt_out_txt,nmt_att_score=self.nmt(src_txt,trg_txt,teacher_forcing_ratio)
        if mode=='normal':
            mel_out,stop_targets,tts_att_score=self.tts(nmt_out_txt.argmax(-1),trg_mel)

        elif mode=='osamura':
            mel_out,stop_targets,tts_att_score=self.tts(F.softmax(nmt_out_txt,-1),trg_mel)

        return nmt_out_txt,mel_out,stop_targets,nmt_att_score,tts_att_score

    def get_path(self):
        return "_".join(hp.rnn)+"_nmt_hid"+str(hp.nmt_h_size)+"_depth"+str(len(hp.rnn))+"/"

    def load(self,args):
        nmt_path=os.path.join("./exp","NMT",args.src_lang+"2"+args.trg_lang,self.nmt.get_path(),args.segment)
        self.nmt.load_state_dict(torch.load(os.path.join(nmt_path,"CHAMP","NMT")))
        tts_path=os.path.join("./exp","Tacotron",args.trg_lang,self.tts.get_path(),args.segment)
        self.tts.load_state_dict(torch.load(os.path.join(tts_path,"CHAMP","Tacotron")))
