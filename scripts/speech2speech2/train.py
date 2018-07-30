import hyperparams as hp
import sys,os,argparse,torch
from torch import nn,optim
from torch.nn import functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../../')
from tqdm import tqdm

from util.show import *
from util.new_data import get_dataset,DataLoader, is_cfn, get_param_size
from new_models.Speech2Speech.S2S_network import Speech2Speech2
import base

class train(base.base):
    def __init__(self):
        self.name="SpeechToSpeech"
        super().__init__()

        self.model=Speech2Speech2(len(self.dataset.src_id2word),len(self.dataset.trg_id2word))
        self.path=os.path.join(self.path,self.model.get_path(),self.args.segment)

        if not os.path.exists(self.path):
            os.makedirs(self.path+"/CHAMP")

        self.ce_criterion=nn.CrossEntropyLoss(ignore_index=0)
        self.mse_criterion=nn.MSELoss()
        self.l1_criterion=nn.L1Loss()
        self.stop_criterion=nn.BCEWithLogitsLoss()

        self.lr=hp.lr
        self.optimizer=optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_loss=list()
        self.dev_loss=list()
        self.phase=0
        self.load()

    def load(self,):
        if super().__load__():
            if os.path.exists(self.path+"/train_loss"):
                with open(self.path+"/train_loss") as f:
                    for i in f.readlines():
                        i=i.strip()
                        self.train_loss.append(float(i))

            if os.path.exists(self.path+"/dev_loss"):
                with open(self.path+"/dev_loss") as f:
                    for i in f.readlines():
                        i=i.strip()
                        self.dev_loss.append(float(i))

            self.epoch=len(self.dev_loss)
            self.timesteps=len(self.train_loss)*1000
            print("Epoch %d train loss %3f dev L1 loss %2f"%(self.epoch,self.train_loss[-1],self.dev_loss[-1]))
        else:
            self.model.load(self.args)
    def save(self,):
        super().__save__()
        with open(self.path+"/train_loss","w") as f:
            for i in self.train_loss:
                f.write(str(float(i))+'\n')

        with open(self.path+"/dev_loss","w") as f:
            for i in self.dev_loss:
                f.write(str(float(i))+'\n')

        if min(self.dev_loss)==self.dev_loss[-1]:
            torch.save(self.model.state_dict(),os.path.join(self.path,"CHAMP",self.name))
            print("Saving model at ",self.epoch)
            if self.dev_loss[-1] >= hp.stop_learning:
                self.epoch=hp.epochs

        show_loss(self.train_loss,self.path+"/train_loss.pdf",self.ce_criterion._get_name())
        show_loss(self.dev_loss,self.path+"/dev_loss.pdf","L1 loss")

    def eval(self,txts,outs,scores,refs):
        for txt,out,score,ref in zip(txts,outs,scores,refs):
            sen=self.dataset.i2w4eval(txt,self.args.src_lang)
            txt=self.dataset.i2w(txt,self.args.src_lang)
            show_attention(score,sen,os.path.join(self.path,"Attention",self.state,str(self.epoch)),label_x=None,label_y=txt.split(" "))
            show_spectrogram(ref,os.path.join(self.path,"Spectrogram",self.state,str(self.epoch),"ref/"),sen)
            show_spectrogram(out,os.path.join(self.path,"Spectrogram",self.state,str(self.epoch),"out/"),sen)


    def develop(self):
        super().__develop__()
        loss=0
        for i, (src_mel,src_txt,trg_txt,trg_mel,stop_targets) in self.pbar:
            with torch.no_grad():
                trans_out,mel_out,stop_targets,att_score,asr_txt,nmt_txt=self.model(src_mel.cuda(),src_txt.cuda(),trg_txt.cuda(),trg_mel.cuda(),self.teacher_forcing_ratio,self.phase)
            length=min(mel_out.size(-1),trg_mel.size(-1))
            loss+=torch.nn.functional.l1_loss(mel_out[:,:,:length].cpu(),trg_mel[:,:,:length])
            if i<15:
                self.eval(trg_txt.numpy(),mel_out.cpu().numpy(),att_score.cpu().numpy(),trg_mel.numpy())
            self.pbar.set_description(f'Dev L1_loss: {loss/(i+1):.4f}')
        self.dev_loss.append(loss/len(self.dataloader))

    def test(self):
        super().__test__()
        loss=0
        with open(os.path.join(self.path,"result"),"w") as f:
            for i, (src_mel,src_txt,trg_txt,trg_mel,stop_targets) in self.pbar:
                with torch.no_grad():
                    trans_out,mel_out,stop_targets,att_score,asr_txt,nmt_txt=self.model(src_mel.cuda(),src_txt.cuda(),trg_txt.cuda(),trg_mel.cuda(),self.teacher_forcing_ratio,self.phase)
                length=min(mel_out.size(-1),trg_mel.size(-1))
                loss+=torch.nn.functional.l1_loss(mel_out[:,:,:length].cpu(),trg_mel[:,:,:length])
                if i<15:
                    self.eval(trg_txt.numpy(),mel_out.cpu().numpy(),att_score.cpu().numpy(),trg_mel.numpy())
                self.pbar.set_description(f'Test L1_loss: {loss/(i+1):.4f}')
            f.write("L1 loss:"+str(loss/(len(self.dataloader))))


    def train(self):
        super().__train__()
        train_l1_loss=0
        train_ce_loss=0
        train_mel_loss=0
        for i, (src_mel,src_txt,trg_txt,trg_mel,stop_targets) in self.pbar:
            #import pdb; pdb.set_trace()
            trans_out,mel_out,stop_outputs,att_score,asr_txt,nmt_txt=self.model(src_mel.cuda(),src_txt.cuda(),trg_txt.cuda(),trg_mel.cuda(),self.teacher_forcing_ratio,self.phase)
            if self.phase in {0}:
                with torch.no_grad():
                    tts_enc=self.model.taco.encode(trg_txt.cuda())
                loss=self.mse_criterion(trans_out,tts_enc)
                train_mel_loss+=torch.nn.functional.l1_loss(mel_out.cpu(),trg_mel)
                train_l1_loss+=loss.item()
            else:

                mel_loss=self.l1_criterion(mel_out,trg_mel.cuda())
                stop_loss=self.stop_criterion(stop_outputs.squeeze(1),stop_targets.cuda())
                loss=mel_loss+stop_loss
                train_mel_loss+=mel_loss.item()
            #import pdb; pdb.set_trace()
            ce_loss=self.ce_criterion(nmt_txt.transpose(1,2),trg_txt.cuda())#+self.ce_criterion(asr_txt.transpose(1,2),src_txt.cuda())
            train_ce_loss+=ce_loss.item()
            loss+=ce_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
            self.model.zero_grad()
            self.optimizer.zero_grad()

            #self.teacher_forcing_ratio=train_loss/(i+1)
            if self.timesteps % 1000  == 0 and self.timesteps>1:
                self.train_loss.append(train_mel_loss/(i+1))
            self.timesteps+=1
            self.pbar.set_description(f'Epoch {self.epoch} ph {self.phase} l1: {train_l1_loss/(i+1):.4f} mel: {train_mel_loss/(i+1):.4f} ce: {train_ce_loss/(i+1):.1f} ')


    def adjust_learning_rate(self):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

        if 0<=self.epoch<=10 :
            self.teacher_forcing_ratio=1
            self.lr=0.001
            self.phase=0
        elif 10< self.epoch <= 20:
            self.teacher_forcing_ratio=1
            self.lr=0.0005
            self.phase=0
        elif 20< self.epoch <=40 :
            self.teacher_forcing_ratio=1
            self.lr=0.0001
            self.phase=1
        elif 40< self.epoch :
            self.teacher_forcing_ratio=0.7
            self.lr=0.0001
            self.phase=1

        for param_group in self.optimizer.param_groups:
            param_group['lr'] =self.lr


if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    Speech2Speech=train()
    #InterSpeech.load()
    Speech2Speech()
