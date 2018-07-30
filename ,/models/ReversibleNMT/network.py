import random,sys,os,torch
from modules.Sequential_modules import *
from modules.Basic_modules import CustomEmbedding
from modules.Attention_modules import *

from torch import nn
import models.NMT.hyperparams as hp


class RNMT(nn.Module):
    def __init__(self,src_vocab,trg_vocab):
        super(RNMT, self).__init__()
        self.src_emb=CustomEmbedding(src_vocab,hp.embed_size,hp.embed_drop)
        self.trg_emb=CustomEmbedding(trg_vocab,hp.embed_size,hp.embed_drop)
        self.encoder=BaseEncoder(hp.embed_size,hp.enc_h_size,hp.enc_drop,hp.bidirectional,hp.rnn)
        self.att_fn=None#Attention(hp.att_type,hp.enc_h_size)
        self.decoder=StandardDecoder(hp.embed_size,hp.dec_h_size,trg_vocab,hp.dec_drop)

    def get_path(self):
        return hp.dec_type+"_"+"_".join(hp.rnn)+"_"+hp.att_type+"/"+"emb"+str(hp.embed_size)+"_hid"+str(hp.enc_h_size)+"_depth"+str(len(hp.rnn))+"/"

    def __call__(self,src,trg=None,teacher_forcing_ratio=1.,mode="normal"):
        #to GPU
        self.reverse()
        enc_out=self.encode(src)
        out_txt,att_score,_=self.decode(trg,enc_out,teacher_forcing_ratio,mode)
        return out_txt,att_score

    def encode(self,txt):
        #import pdb; pdb.set_trace()
        enc_input=self.src_emb(txt)
        enc_out=self.encoder(enc_input)
        return enc_out

    def decode(self,txt,enc_out,teacher_forcing_ratio,mode):
        self.decoder.init(enc_out)
        out_txt=list()
        att_score=list()
        out_context=list()
        dec_input=torch.ones((enc_out.size(0))).type(torch.cuda.LongTensor)
        dec_input=self.trg_emb(dec_input)

        output_len=txt.size(1) if txt is not None else 100
        for ii in range(output_len):

            dec_out,att_weights,context=self.decoder(dec_input,self.att_fn)
            out_txt.append(dec_out)
            att_score.append(att_weights)
            out_context.append(context)

            if txt is not None and random.random()<teacher_forcing_ratio:
                dec_input=self.trg_emb(txt[:,ii])
            else:
                _,dec_input=dec_out.topk(1)
                if not self.training:
                    if int(dec_input) ==hp.EOS:# EOS=1
                        break
                #import pdb; pdb.set_trace()
                if mode =="normal":
                    dec_input=self.trg_emb(dec_out.argmax(-1))
                elif mode =="osamura":
                    dec_input=self.trg_emb(F.softmax(dec_out,-1))
                #dec_input=self.trg_emb.emb(nn.functional.softmax(w,-1))


        out_txt    = torch.stack(out_txt, 1)
        att_score  = torch.stack(att_score,1)
        out_context    = torch.stack(out_context,1)
        return out_txt,att_score,out_context

    def reverse(self):
        import pdb; pdb.set_trace()
        for c in self.encoder.children():
            for p in c.parameters():
                pass
            pass
        pass
