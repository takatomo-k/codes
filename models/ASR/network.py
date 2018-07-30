#-*- coding: utf-8 -*-
import random,sys,os,torch
from torch import nn
from modules.Sequential_modules import *
from modules.Basic_modules import *
from modules.Attention_modules import *
import models.ASR.hyperparams as hp
from torch.nn import functional as F

class ASR(nn.Module):
    def __init__(self,vocab_size):
        super(ASR, self).__init__(in_size,emg_drop,enc_in,enc_hidden,enc_drop,enc_bid,enc_rnn,att_type,dec_in,dec_hidden,dec_drop,out_size)
        self.prenet=MLP(in_size,enc_in,emb_drop)
        self.encoder=BaseEncoder(enc_in,enc_hidden,enc_drop,enc_bid,enc_rnn)
        self.att_fn=Attention(att_type,enc_hidden)
        self.decoder=LuongDecoder(dec_in,dec_hidden,out_size,dec_drop)
        self.emb=CustomEmbedding(vocab_size,dec_in,embed_drop)

    def __call__(self,mel,txt=None,teacher_forcing_ratio=1.,mode="normal"):
        enc_out=self.encode(mel,skip_step=2)
        out_txt,att_score,_=self.decode(txt,enc_out,teacher_forcing_ratio,mode)
        return out_txt,att_score

    def encode(self,mel,skip_step=2):
        enc_input=self.prenet(mel)
        enc_out=self.encoder(enc_input,skip_step=skip_step)
        return enc_out

    def decode(self,txt,enc_out,teacher_forcing_ratio,mode):
        #import pdb; pdb.set_trace()
        out_txt=list()
        att_score=list()
        out_context=list()
        dec_input=torch.ones((enc_out.size(0))).type(torch.cuda.LongTensor)
        dec_input=self.emb(dec_input)

        output_len=txt.size(1) if txt is not None else 100
        self.decoder.init(enc_out)
        for ii in range(output_len):
            dec_out,attn_weights,context=self.decoder(dec_input,self.att_fn)
            #import pdb; pdb.set_trace()
            if txt is not None and random.random()<teacher_forcing_ratio:
                dec_input=self.emb(txt[:,ii])
            else:
                _,dec_input=dec_out.topk(1)
                if not self.training:
                    if int(dec_input) ==hp.EOS:# EOS=1
                        break
                dec_input=self.emb(dec_input[0])
            out_txt.append(dec_out)
            att_score.append(attn_weights)
            out_context.append(context)

        #import pdb; pdb.set_trace()
        out_txt    = torch.stack(out_txt,1)
        att_score  = torch.stack(att_score,1)
        out_context= torch.stack(out_context,1)

        return out_txt,att_score,out_context
