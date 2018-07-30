#-*- coding: utf-8 -*-
import random,sys,os,torch
from torch import nn
from modules.Sequential_modules import *
from modules.Basic_modules import *
from modules.Attention_modules import *
from models.ASR.network import ASR
from models.NMT.network import NMT
import models.InterSpeech.hyperparams as hp

class InterSpeech(nn.Module):
    def __init__(self,src_vocab_size,trg_vocab_size):
        super(InterSpeech, self).__init__()
        self.asr=ASR(src_vocab_size)
        self.nmt=NMT(src_vocab_size,trg_vocab_size)
        self.transcoder=BaseEncoder(hp.asr_h_size,hp.nmt_h_size,hp.drop,hp.bidirectional,hp.rnn)

    def __call__(self,mel,src_txt=None,trg_txt=None,teacher_forcing_ratio=1.,phase=2):
        #self.flatten_parameters()
#        with torch.set_grad_enabled(self.training):
        #import pdb; pdb.set_trace()
        with torch.set_grad_enabled(phase in {1,2} and self.training):
            asr_enc_out=self.asr.encode(mel,skip_step=2)
            asr_out_txt,asr_att_score,context=self.asr.decode(src_txt,asr_enc_out,teacher_forcing_ratio)

        with torch.set_grad_enabled(self.training):
            trans_out =self.transcoder(context)

        with torch.set_grad_enabled(phase ==2 and self.training):
            out_txt,att_score,_=self.nmt.decode(trg_txt,trans_out,teacher_forcing_ratio)

        return trans_out,out_txt,att_score


    def get_path(self):
        return "_".join(hp.rnn)+"_asr_hid"+str(hp.asr_h_size)+"_nmt_hid"+str(hp.nmt_h_size)+"_depth"+str(len(hp.rnn))+"/"

    def load(self,args):

        asr_path=os.path.join("./exp","ASR",args.src_lang,self.asr.get_path(),args.segment)
        self.asr.load_state_dict(torch.load(os.path.join(asr_path,"CHAMP","ASR")))
        nmt_path=os.path.join("./exp","NMT",args.src_lang+"2"+args.trg_lang,self.nmt.get_path(),args.segment)
        self.nmt.load_state_dict(torch.load(os.path.join(nmt_path,"CHAMP","NMT")))

    def encode_wo_dump(self,mel,src_txt=None,trg_txt=None,teacher_forcing_ratio=1.):
        with torch.no_grad():
            asr_enc_out=self.asr.encode(mel,skip_step=2)
            asr_out_txt,asr_att_score,context=self.asr.decode(src_txt,asr_enc_out,teacher_forcing_ratio)
            trans_out =self.transcoder(context)
            out_txt,att_score,result=self.nmt.decode(trg_txt,trans_out,teacher_forcing_ratio)

        return asr_out_txt,out_txt,result

    def encode(self,mel,src_txt=None,trg_txt=None,teacher_forcing_ratio=1.):

        asr_enc_out=self.asr.encode(mel,skip_step=2)
        asr_out_txt,asr_att_score,context=self.asr.decode(src_txt,asr_enc_out,teacher_forcing_ratio)
        trans_out =self.transcoder(context)
        out_txt,att_score,_=self.nmt.decode(trg_txt,trans_out,teacher_forcing_ratio)

        return trans_out,out_txt,att_score
