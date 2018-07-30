#-*- coding: utf-8 -*-
import random,sys,os,torch
from torch import nn
import torch.functionals as F
from modules.Sequential_modules import *
from modules.Basic_modules import *
from modules.Attention_modules import *
from models.InterSpeech.network import InterSpeech
from models.Tacotron.Taco_network import Tacotron
from models.ASR.network import ASR
from models.NMT.network import NMT

import new_models.Speech2Speech.hyperparams as hp
class Speech2Speech(nn.Module):
    def __init__(self,src_vocab_size,trg_vocab_size):
        super(Speech2Speech, self).__init__()
        #self.inter_speech=InterSpeech(src_vocab_size,trg_vocab_size)
        #self.transcoder=BaseEncoder(hp.nmt_h_size,hp.tts_h_size,hp.drop,hp.bidirectional,hp.rnn)
        self.asr=ASR(src_vocab)
        self.nmt=NMT(src_vocab,trg_vocab)
        self.taco=Tacotron(trg_vocab_size)

    def pipe(self,mel,src_txt=None,trg_txt=None,trg_mel=None,teacher_forcing_ratio=1.):
        asr_out,asr_att,_=self.asr(mel,src_txt)
        asr_out=asr_out.argmax(-1)
        nmt_out,nmt_att,_=self.nmt(asr_out,trg_txt)
        mel_out,stop_targets,att_score=self.taco.decode(nmt_out,tts_ref,teacher_forcing_ratio)
        return mel_out,stop_targets,att_score

    def osamura(self,mel,src_txt=None,trg_txt=None,trg_mel=None,teacher_forcing_ratio=1.):
        asr_out,asr_att,_=self.asr(mel,src_txt)
        asr_out=F.softmax(asr_out,-1)
        nmt_out,nmt_att,_=self.nmt(asr_out,trg_txt)
        nmt_out=F.softmax(nmt_out,-1)
        mel_out,stop_targets,att_score=self.taco(nmt_out,trg_mel,teacher_forcing_ratio)
        return mel_out,stop_targets,att_score

    def __call__(self,mel,src_txt=None,trg_txt=None,trg_mel=None,teacher_forcing_ratio=1.,phase=2):
        #import pdb; pdb.set_trace()
        with torch.set_grad_enabled(phase ==2 and self.training):
            _,out_txt,context=self.inter_speech.encode_wo_dump(mel,src_txt,trg_txt,teacher_forcing_ratio)
            tts_enc    =self.taco.encode(torch.argmax(out_txt,-1))
            tts_ref,_,_=self.taco.decode(tts_enc,None,teacher_forcing_ratio)

        with torch.set_grad_enabled(self.training):
            trans_out=self.transcoder(context)

        with torch.set_grad_enabled(phase in {1,2} and self.training):
            mel_out,stop_targets,att_score=self.taco.decode(trans_out,tts_ref,teacher_forcing_ratio)

        return trans_out,mel_out,stop_targets,att_score,tts_enc,tts_ref


    def get_path(self):
        return "_".join(hp.rnn)+"_asr_hid"+str(hp.tts_h_size)+"_nmt_hid"+str(hp.nmt_h_size)+"_depth"+str(len(hp.rnn))+"/"

    def load(self,args):
        is_path=os.path.join("./exp","InterSpeech",args.src_lang+"2"+args.trg_lang,self.inter_speech.get_path(),args.segment)
        self.inter_speech.load_state_dict(torch.load(os.path.join(is_path,"CHAMP","InterSpeech")))

class Speech2Speech2(nn.Module):
    def __init__(self,src_vocab_size,trg_vocab_size):
        super(Speech2Speech2, self).__init__()
        self.inter_speech=InterSpeech(src_vocab_size,trg_vocab_size)
        self.transcoder=BaseEncoder(hp.nmt_h_size,hp.tts_h_size,hp.drop,hp.bidirectional,hp.rnn)
        self.taco=Tacotron(trg_vocab_size)

    def __call__(self,mel,src_txt=None,trg_txt=None,trg_mel=None,teacher_forcing_ratio=1.,phase=2):
        #import pdb; pdb.set_trace()

        with torch.set_grad_enabled(self.training):
            asr_txt,nmt_txt,context=self.inter_speech.encode_wo_dump(mel,src_txt,trg_txt,teacher_forcing_ratio)
            trans_out=self.transcoder(context)

        with torch.set_grad_enabled(phase ==1 and self.training):
            mel_out,stop_targets,att_score=self.taco.decode(trans_out,trg_mel,teacher_forcing_ratio)

        return trans_out,mel_out,stop_targets,att_score,asr_txt,nmt_txt


    def get_path(self):
        return "_".join(hp.rnn)+"_asr_hid"+str(hp.tts_h_size)+"_nmt_hid"+str(hp.nmt_h_size)+"_depth"+str(len(hp.rnn))+"/"

    def load(self,args):
        is_path=os.path.join("./exp","InterSpeech",args.src_lang+"2"+args.trg_lang,self.inter_speech.get_path(),args.segment)
        self.inter_speech.load_state_dict(torch.load(os.path.join(is_path,"CHAMP","InterSpeech")))
