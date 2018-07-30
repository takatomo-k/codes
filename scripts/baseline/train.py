import hyperparams as hp
from torch import nn,optim
from tqdm import tqdm

import sys,os,argparse,torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../../')

from util.show import *
from util.new_data import get_dataset,DataLoader, asr_cfn, get_param_size
from util.evaluations import BLEUp1
from new_models.Asr.asr_network import ASR,CustomASR
import base
class train(base.base):
    def __init__(self):
        self.name="Baseline"
        super().__init__()
        self.model=ASR(len(self.dataset.trg_id2word))
        self.path=os.path.join(self.path,self.model.get_path(),self.args.segment)
        if not os.path.exists(os.path.join(self.path,"CHAMP")):
            os.makedirs(os.path.join(self.path,"CHAMP"))

        self.criterion=nn.CrossEntropyLoss(ignore_index=0)
        self.lr=hp.lr
        self.optimizer=optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_loss=list()
        self.dev_loss=list()
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
            print("Epoch %d train loss %3f dev loss %4f"%(self.epoch,self.train_loss[-1],self.dev_loss[-1]))

    def save(self,):
        super().__save__()
        with open(self.path+"/train_loss","w") as f:
            for i in self.train_loss:
                f.write(str(float(i))+'\n')

        with open(self.path+"/dev_loss","w") as f:
            for i in self.dev_loss:
                f.write(str(float(i))+'\n')

        if max(self.dev_loss)==self.dev_loss[-1]:
            torch.save(self.model.state_dict(),os.path.join(self.path,"CHAMP",self.name))
            print("Saving model at ",self.epoch)
            if self.dev_loss[-1] <= hp.stop_learning:
                self.epoch=self.args.epochs

        show_loss(self.train_loss,os.path.join(self.path,"train_loss.pdf"),self.criterion._get_name())
        show_loss(self.dev_loss,os.path.join(self.path,"dev_loss.pdf"),"BLEU")

    def eval(self,srcs,trgs,outs,scores,dump=False,f=None):
        #import pdb; pdb.set_trace()
        # TODO: *argsで書き直す
        loss=0
        for i,(src,ref,out,score) in enumerate(zip(srcs,trgs,outs,scores)):
            src_sen=self.dataset.i2w4eval(src,self.args.src_lang)
            ref_sen=self.dataset.i2w4eval(ref,self.args.trg_lang)
            out_sen=self.dataset.i2w4eval(np.argmax(out,axis=-1),self.args.trg_lang)
            #import pdb; pdb.set_trace()
            loss+=BLEUp1(out_sen,ref_sen)
            if f is not None:
                f.write(out_sen+"|"+ref_sen+"\n")

            if dump:
                src=self.dataset.i2w(src,self.args.src_lang)
                out=self.dataset.i2w(np.argmax(out,axis=-1),self.args.trg_lang)

                show_attention(score,ref_sen,os.path.join(self.path,self.state,"Attention",str(self.epoch)),label_x=src.split(),label_y=out.split())
        return loss/(i+1)

    def develop(self):
        super().__develop__()
        loss=0
        for i, (mel,src,txt) in self.pbar:
            with torch.no_grad():
                out_txt,att_score=self.model(mel.cuda())
                loss+=self.eval(src.numpy(),txt.numpy(),out_txt.cpu().numpy(),att_score.cpu().numpy(),dump=i<15)
            self.pbar.set_description(f'Develop BLEU: {loss/(i+1):.4f}')
        self.dev_loss.append(loss/(len(self.dataloader)))

    def test(self):
        super().__test__()
        loss=0
        with open(os.path.join(self.path,"TestBLEU"),"w") as f:
            for i, (mel,src,txt) in self.pbar:
                with torch.no_grad():
                    out_txt,att_score=self.model(mel.cuda())
                loss+=self.eval(src.numpy(),txt.numpy(),out_txt.cpu().numpy(),att_score.cpu().numpy(),dump=i<15,f=f)
                self.pbar.set_description(f'Test BLEU: {loss/(i+1):.4f}')
            f.write(str(loss/len(self.dataloader)))

    def train(self):
        super().__train__()
        train_loss=0
        train_wer=0
        for i, (mel,src,txt) in self.pbar:
            #import pdb; pdb.set_trace()
            out_txt,att_score=self.model(mel.cuda(),txt.cuda(),self.teacher_forcing_ratio)
            loss=self.criterion(out_txt.transpose(1,2),txt.cuda())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()

            self.model.zero_grad()
            self.optimizer.zero_grad()
            train_loss+=loss.item()

            if self.timesteps % 1000  == 0 and self.timesteps>1:
                self.train_loss.append(train_loss/(i+1))
            self.timesteps+=1

            #if len(self.dataloader)-i<10:
            #    train_wer+=self.eval(src.numpy(),txt.numpy(),out_txt.detach().cpu().numpy(),att_score.detach().cpu().numpy(),dump=len(self.dataloader)-i==1)
            #    self.pbar.set_description(f'Epoch {self.epoch} tr {self.teacher_forcing_ratio:.2f} lr {self.lr:.4f} loss {train_loss/(i+1):.3f} BLEU: {train_wer/(10-len(self.dataloader)+i):.1f}')
            #else:
            self.pbar.set_description(f'Epoch {self.epoch} tr {self.teacher_forcing_ratio:.2f} lr {self.lr:.4f} loss {train_loss/(i+1):.3f}')

    def adjust_learning_rate(self):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if 0<= self.epoch <= 10:
            self.lr=0.001
            self.teacher_forcing_ratio=1.

        elif 10< self.epoch <= 20:
            self.lr=0.0005
            self.teacher_forcing_ratio=1

        elif 20< self.epoch :
            self.lr=0.0001
            self.teacher_forcing_ratio=1.

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    asr=train()
    asr()
