import hyperparams as hp
from torch import nn,optim
from tqdm import tqdm

import sys,os,argparse,torch,io
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../../')

from util.show import *
from util.data import get_dataset,DataLoader, collate_fn, get_param_size,inv_spectrogram,save_wav
from util.evaluations import wer
from models.Tacotron.network import Vocoder
import base
class train(base.base):
    def __init__(self,src_lang,trg_lang,segment,log_dir,batch_size,n_workers,epochs):
        self.name="Vocoder"
        self.data_type="Speech"
        super().__init__(src_lang,trg_lang,segment,log_dir,batch_size,n_workers,epochs)
        self.model=Vocoder()
        self.path=os.path.join(self.path,self.model.get_path())
        if not os.path.exists(os.path.join(self.path,"CHAMP")):
            os.makedirs(os.path.join(self.path,"CHAMP"))
        self.criterion=nn.MSELoss()#nn.L1Loss()
        self.lr=hp.lr
        self.optimizer=optim.Adam(self.model.parameters(), lr=hp.lr)
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
            print("Epoch %d train loss %3f dev loss %2f"%(self.epoch,self.train_loss[-1],self.dev_loss[-1]))

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
            

        show_loss(self.train_loss,self.path+"/train_loss.pdf",self.criterion._get_name())
        show_loss(self.dev_loss,self.path+"/dev_loss.pdf","WER")

    def eval(self,):
        with torch.no_grad():
            linear_out=self.model(self.data['trg_mel'].cuda())
            loss=torch.nn.functional.smooth_l1_loss(linear_out.cpu(),self.data['trg_linear'])
            self.total_loss+=loss

        return loss,linear_out.cpu(),self.data['trg_linear'], "SL1 %0.4f "%(self.total_loss/(self.i+1))

    def train(self):
        src,trg=self.data['trg_mel'].cuda(),self.data['trg_linear'].cuda()
        linear_out = self.model(src)
        loss=self.criterion(linear_out,trg)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optimizer.step()
        self.model.zero_grad()
        self.optimizer.zero_grad()
        self.total_loss+=loss.item()

        return "Epoch %d tr %0.1f SL1_loss %0.4f"%(self.epoch,self.teacher_forcing_ratio,self.total_loss/(self.i+1))

if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_lang', type=str, help='src language')
    parser.add_argument('--segment', type=str, help='src language',default="word")
    parser.add_argument('--trg_lang', type=str, help='src language', default=None)
    parser.add_argument('--restore_epoch', type=int, help='src language', default=None)
    parser.add_argument('--log_dir', type=str, help='src language', default="./exp")
    parser.add_argument('--batch_size', type=int, help='Batch size', default=16)
    parser.add_argument('-n','--n_workers', type=int, help='Batch size', default=8)
    parser.add_argument('--epochs', type=int, help='Batch size', default=15)
    args = parser.parse_args()

    if not os.path.exists(args.log_dir+"/Vocoder/"):
        os.makedirs(args.log_dir+"/Vocoder/")


    log_f=open(args.log_dir+"/Vocoder/log","w")
    log_f.write(args.src_lang+"\t"+"START"+"\n")
    tacotron=train(args.src_lang,None,args.segment,args.log_dir,args.batch_size,args.n_workers,args.epochs)
    result=tacotron()
    log_f.write(args.src_lang+"\t"+result+"END"+"\n")
