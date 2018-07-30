#-*- coding: utf-8 -*-
import random,sys,os,torch
from torch import nn
import models.Tacotron.hyperparams as hp
from modules.Tacotron_modules import *
from modules.Sequential_modules import BaseEncoder

class CHBGEncoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, embed_size,hidden_size):
        """

        :param embed_size: dimension of embedding
        """
        super(CHBGEncoder, self).__init__()
        self.prenet = Prenet(hp.embed_size, hp.hidden_size * 2, hp.hidden_size)
        self.cbhg = CBHG(hp.hidden_size)

    def forward(self, input_):

        input_ = torch.transpose(input_,1,2)
        prenet = self.prenet.forward(input_)
        memory = self.cbhg.forward(prenet)
        return memory

class MelDecoder(nn.Module):
    """
    Decoder
    """
    def __init__(self,hidden_size,num_mels,outputs_per_step):
        super(MelDecoder, self).__init__()
        self.prenet = Prenet(num_mels,hidden_size * 2, hidden_size)
        self.attn_decoder = AttentionDecoder(hidden_size * 2,num_mels,outputs_per_step)

    def forward(self, decoder_input, memory,teacher_forcing_ratio):

        # Initialize hidden state of GRUcells
        attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.inithidden(memory.size(0))
        outputs = list()
        att_score= list()
        stop_targets=list()
        # Training phase
        if self.training and decoder_input is not None:
            # Prenet
            dec_input = self.prenet.forward(decoder_input)
            timesteps = dec_input.size()[2] // hp.outputs_per_step

            # [GO] Frame
            prev_output = dec_input[:, :, 0]

            for i in range(timesteps):

                prev_output,stop, attn_hidden, gru1_hidden, gru2_hidden,attn_weights = self.attn_decoder.forward(prev_output, memory,
                                                                                             attn_hidden=attn_hidden,
                                                                                             gru1_hidden=gru1_hidden,
                                                                                             gru2_hidden=gru2_hidden)

                outputs.append(prev_output)
                att_score.append(attn_weights)
                stop_targets.append(stop)
                if random.random() < teacher_forcing_ratio:
                    # Get spectrum at rth position
                    prev_output = dec_input[:, :, i * hp.outputs_per_step]
                else:
                    # Get last output
                    prev_output = self.prenet.forward(prev_output[:, :, -1].unsqueeze(2)).squeeze(2)

            # Concatenate all mel spectrogram

        else:
            # [GO] Frame
            #import pdb; pdb.set_trace()
            prev_output = torch.zeros((memory.size(0),1,hp.num_mels)).cuda()

            for i in range(hp.max_iters):
                prev_output = self.prenet.forward(prev_output)
                prev_output = prev_output[:,:,0]
                prev_output, stop,attn_hidden, gru1_hidden, gru2_hidden,attn_weights = self.attn_decoder.forward(prev_output, memory,
                                                                                         attn_hidden=attn_hidden,
                                                                                         gru1_hidden=gru1_hidden,
                                                                                         gru2_hidden=gru2_hidden)

                outputs.append(prev_output)
                att_score.append(attn_weights)
                stop_targets.append(stop)
                if torch.max(stop).item()>0.5:
                    break
                prev_output = prev_output[:, :, -1].unsqueeze(2)

        #import pdb; pdb.set_trace()
        outputs = torch.cat(outputs, 2)
        att_score= torch.cat(att_score, 2)
        stop_targets=torch.cat(stop_targets,2)
        return outputs,stop_targets,att_score

class PostProcessingNet(nn.Module):
    """
    Post-processing Network
    """
    def __init__(self,hidden_size,num_mels,num_freq):
        super(PostProcessingNet, self).__init__()
        self.postcbhg = CBHG(hidden_size,
                             K=8,
                             projection_size=num_mels,
                             is_post=True)
        self.linear = SeqLinear(hidden_size * 2,
                                num_freq)

    def forward(self, input_):
        out = self.postcbhg.forward(input_)
        out = self.linear.forward(torch.transpose(out,1,2))

        return out

class Tacotron(nn.Module):
    """
    End-to-end Tacotron Network
    """
    def __init__(self,vocab_size):
        super(Tacotron, self).__init__()
        self.embed   = nn.Embedding(vocab_size,hp.embed_size)
        if hp.enc_type=="CHBG":
            self.encoder = CHBGEncoder(hp.embed_size,hp.hidden_size)
        elif hp.enc_type=="BASIC":
            self.encoder = BaseEncoder(hp.embed_size,hp.hidden_size*2,hp.enc_drop,hp.bidirectional,hp.enc_rnn)
        #self.encoder = EncoderRNN(hp.embed_size,hp.hidden_size,hp.n_layers,hp.dropout)
        #import pdb; pdb.set_trace()
        # TODO: Fix tacotron net setting
        self.decoder1 = MelDecoder(hp.hidden_size,hp.num_mels,hp.outputs_per_step)
        #self.decoder2 = PostProcessingNet(hp.hidden_size,hp.num_mels,hp.num_freq)

    def forward(self, txt, mel_input=None,teacher_forcing_ratio=1.):
        #import pdb; pdb.set_trace()
        if isinstance(txt,torch.cuda.FloatTensor):
            if txt.dim()>2:
                embed=txt.bmm(self.embed.weight.unsqueeze(0).repeat(txt.size(0),1,1))
            else:
                embed=txt.mm(self.embed.weight)
        else:
            embed=self.embed(txt)


        memory = self.encoder.forward(embed)
        mel_out,stop_targets,att_score = self.decoder1.forward(mel_input, memory,teacher_forcing_ratio)
        #linear_output = self.decoder2.forward(mel_out)

        return mel_out, stop_targets,att_score,#linear_output
    def encode(self,txt):
        txt=self.embed(txt)
        memory = self.encoder.forward(txt)
        return memory
    def decode(self,memory,mel_input,teacher_forcing_ratio):
        mel_out,stop_targets,att_score = self.decoder1.forward(mel_input, memory,teacher_forcing_ratio)
        return mel_out,stop_targets,att_score
    def get_path(self):
        return hp.enc_type+"_"+hp.att_type+"/"+"emb"+str(hp.embed_size)+"_hid"+str(hp.hidden_size)+"_depth"+str(hp.n_layers)+"/"

class Vocoder(nn.Module):
    """
    End-to-end Tacotron Network
    """
    def __init__(self):
        super(Vocoder, self).__init__()
        self.decoder2 = PostProcessingNet(hp.hidden_size,hp.num_mels,hp.num_freq)
        self.drop =nn.Dropout(0.3)
    def forward(self,mel_input):
        linear_output = self.decoder2.forward(self.drop(mel_input))
        return linear_output

    def get_path(self):
        return hp.voc_type
