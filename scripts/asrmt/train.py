import hyperparams as hp
import sys,os,argparse,torch
from torch import nn,optim
from torch.nn import functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../../')
from tqdm import tqdm

from util.show import *
from util.evaluations import BLEUp1
from models.ASRMT.network import ASRMT
import base

class train(base.base):
    def __init__(self):
        self.name="ASRMT"
        self.data_type="SpeechText"
        super().__init__()
        #import pdb; pdb.set_trace()
        self.model=ASRMT(len(self.dataset.src_id2word),len(self.dataset.trg_id2word))
        self.path=os.path.join(self.path,self.model.get_path(),self.args.segment)
        if not os.path.exists(self.path):
            os.makedirs(self.path+"/CHAMP")
        self.ce_criterion=nn.CrossEntropyLoss(ignore_index=0)
        self.l1_criterion=nn.SmoothL1Loss()
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
                        try:
                            self.dev_loss.append(float(i))
                        except:
                            #import pdb; pdb.set_trace()
                            pass
            self.epoch=len(self.dev_loss)
            self.timesteps=len(self.train_loss)*1000
            #print("Epoch %d train loss %3f dev BLEUp1 %2f"%(self.epoch,self.train_loss[-1],max(self.dev_loss)))
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

        if max(self.dev_loss)==self.dev_loss[-1]:
            torch.save(self.model.state_dict(),os.path.join(self.path,"CHAMP",self.name))
            print("Saving model at ",self.epoch)
            if self.dev_loss[-1] >= hp.stop_learning:
                self.epoch=hp.epochs

        show_loss(self.train_loss,self.path+"/train_loss.pdf",self.ce_criterion._get_name())
        show_loss(self.dev_loss,self.path+"/dev_loss.pdf","BLEUp1")

    def eval(self,srcs,trgs,outs,scores,dump=False,f=None):
        #import pdb; pdb.set_trace()
        # TODO: *argsで書き直す
        loss=0
        for i,(src,ref,out,score) in enumerate(zip(srcs,trgs,outs,scores)):
            src_sen=self.dataset.i2w4eval(src,self.args.src_lang)
            ref_sen=self.dataset.i2w4eval(ref,self.args.trg_lang)
            out_sen=self.dataset.i2w4eval(np.argmax(out,axis=-1),self.args.trg_lang)
            loss+=BLEUp1(out_sen,ref_sen)
            if f is not None:
                f.write(out_sen+"|"+ref_sen+"\n")
            if dump:
                src=self.dataset.i2w(src,self.args.src_lang)
                out=self.dataset.i2w(np.argmax(out,axis=-1),self.args.trg_lang)

                show_attention(score,ref_sen,os.path.join(self.path,self.state,"Attention",str(self.epoch)),label_x=src.split(),label_y=out.split())
        return loss/(i+1)

    def develop(self):
        super().__test__()
        loss=0
        for i, data in self.pbar:
            with torch.no_grad():
                asr_out_txt,nmt_out_txt,_,nmt_att_score=self.model(data['src_mel'])

            loss+=self.eval(src_txt.numpy(),trg_txt.numpy(),out_txt.cpu().numpy(),nmt_att_score.cpu().numpy(),dump=False)
            self.pbar.set_description(f'Test BLEUp1: {loss/(i+1):.4f}')

    def test(self):
        super().__test__()
        loss=0
        with open(os.path.join(self.path,"result"),"w") as f:
            for i, data in self.pbar:
                with torch.no_grad():
                    asr_out_txt,nmt_out_txt,_,nmt_att_score=self.model(data['src_mel'])

                loss+=self.eval(src_txt.numpy(),trg_txt.numpy(),out_txt.cpu().numpy(),nmt_att_score.cpu().numpy(),dump=i<15,f=f)
                self.pbar.set_description(f'Test BLEUp1: {loss/(i+1):.4f}')
            f.write("BLEUp1:"+str(loss/(len(self.dataloader))))


    def train(self):
        super().__train__()
        train_ce_loss=0
        train_mel_loss=0
        for i, data in self.pbar:
            asr_out_txt,nmt_out_txt,_,_=self.model(data['src_mel'],data['src_txt'].cuda(),data['trg_txt'].cuda(),self.teacher_forcing_ratio)
            asr_loss=self.ce_criterion(asr_out_txt.transpose(1,2),data['src_txt'].cuda())
            nmt_loss=self.ce_criterion(nmt_out_txt.transpose(1,2),data['trg_txt'].cuda())

            loss=asr_loss+nmt_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
            self.model.zero_grad()
            self.optimizer.zero_grad()

            #self.teacher_forcing_ratio=train_loss/(i+1)
            if self.timesteps % 1000  == 0 and self.timesteps>1:
                self.train_loss.append(train_ce_loss/(i+1))
            self.timesteps+=1

            self.pbar.set_description(f'Epoch {self.epoch} lr {self.lr:.4f} mel: {train_mel_loss/(i+1):.4f} ce: {train_ce_loss/(i+1):.4f}')

    def adjust_learning_rate(self):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

        if 0<=self.epoch<=10 :
            self.teacher_forcing_ratio=1
            self.lr=0.001
        elif 10< self.epoch <= 30:
            self.teacher_forcing_ratio=1
            self.lr=0.0005
        elif 30< self.epoch <=40 :
            self.teacher_forcing_ratio=1.0
            self.lr=0.0001
        elif 40< self.epoch <=60 :
            self.teacher_forcing_ratio=1.0
            self.lr=0.0001
        elif 60< self.epoch :
            self.teacher_forcing_ratio=1.0
            self.lr=0.0001

        for param_group in self.optimizer.param_groups:
            param_group['lr'] =self.lr

if __name__ == '__main__':
    Cascade=train()
    Cascade()
