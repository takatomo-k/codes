import hyperparams as hp
import sys,os,argparse,torch
from torch import nn,optim
from torch.nn import functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../../')
from util.show import *
from util.evaluations import BLEUp1
from models.NMT.network import NMT,CustomNMT
import base

class train(base.base):
    def __init__(self,src_lang,trg_lang,segment,log_dir,batch_size,n_workers,epochs):
        self.name="NMT"
        self.data_type="TextText"
        super().__init__(src_lang,trg_lang,segment,log_dir,batch_size,n_workers,epochs)
        self.model=NMT(len(self.dataset.src_vocab[self.segment]),len(self.dataset.trg_vocab[self.segment]))
        self.path=os.path.join(self.path,self.model.get_path(),self.segment)
        if not os.path.exists(self.path+"/CHAMP"):
            os.makedirs(self.path+"/CHAMP")
        self.criterion=nn.CrossEntropyLoss(ignore_index=0)
        self.lr=hp.lr
        self.optimizer=optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_loss=list()
        self.dev_loss=list()
        self.test_result=0
        self.decode_mode='normal'
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

            #self.epoch=len(self.dev_loss)
            #print("Epoch %d train loss %3f dev BLEU %2f"%(self.epoch,self.train_loss[-1],max(self.dev_loss)))

    def save(self,):
        #super().__save__()
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
                self.epoch=self.epochs

        show_loss(self.train_loss,self.path+"/train_loss.pdf",self.criterion._get_name())
        show_loss(self.dev_loss,self.path+"/dev_loss.pdf","BLEUp1")

    def dump(self):
        pass

    def eval(self,mode=None):
        with torch.no_grad():
            out_txt,att_score=self.model(self.data['src']['id_'+self.segment].cuda())
        out_txt=out_txt[0].argmax(-1).cpu().numpy()
        try:
            ref=self.data['trg']['raw_'+self.segment]
        except:
            ref=self.dataset.i2w(self.data['trg']['id_'+self.segment][0].cpu().numpy(),self.segment,self.trg_lang)
        hyp=self.dataset.i2w(out_txt,self.segment,self.trg_lang)
        loss=BLEUp1(hyp,ref)
        self.total_loss+=loss
        return loss,hyp,ref, "BLEU+1 %0.1f "%(self.total_loss/(self.i+1))

    def train(self):
        src,trg=self.data['src']['id_'+self.segment].cuda(),self.data['trg']['id_'+self.segment].cuda()
        out_txt,att_score=self.model(src,trg,self.teacher_forcing_ratio,mode=self.decode_mode)
        loss=self.criterion(out_txt.transpose(1,2),trg)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optimizer.step()
        self.model.zero_grad()
        self.optimizer.zero_grad()
        self.total_loss+=loss.item()
        return "Epoch %d tr %0.1f CE_loss %0.1f"%(self.epoch,self.teacher_forcing_ratio,self.total_loss/(self.i+1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_lang', type=str, help='src language',default=None)
    parser.add_argument('--segment', type=str, help='src language',default="word")
    parser.add_argument('--trg_lang', type=str, help='src language', default=None)
    parser.add_argument('--restore_epoch', type=int, help='src language', default=None)
    parser.add_argument('--log_dir', type=str, help='src language', default="./exp")
    parser.add_argument('--batch_size', type=int, help='Batch size', default=16)
    parser.add_argument('-n','--n_workers', type=int, help='Batch size', default=8)
    parser.add_argument('--epochs', type=int, help='Batch size', default=15)
    args = parser.parse_args()

    src_list=["japanese","english"]
    trg_list=["japanese","english","french","korean"]
    segment_list=["word","sub_word","char"]
    segment_list_ja=["word","sub_word","char","yomi_word","yomi_sub","yomi_char"]

    if not os.path.exists(args.log_dir+"/NMT/"):
        os.makedirs(args.log_dir+"/NMT/")

    log_f=open(args.log_dir+"/NMT/log","w")

    if args.src_lang is None:
        for src_lang in src_list:
            for trg_lang in trg_list:
                if src_lang =="japanese" or trg_lang =="japanese":
                    _list=segment_list_ja
                else:
                    _list=segment_list
                for segment in _list:
                    if src_lang != trg_lang:
                        if src_lang == "japanese" and trg_lang =="french":
                            pass
                        elif src_lang == "english" and trg_lang =="korean":
                            pass
                        else:
                            log_f.write(src_lang+"\t"+trg_lang+"\t"+segment+"START"+"\n")
                            nmt=train(src_lang,trg_lang,segment,args.log_dir,args.batch_size,args.n_workers,args.epochs)
                            result=nmt()
                            log_f.write(src_lang+"\t"+trg_lang+"\t"+segment+"\t"+result+"\t"+"END"+"\n")
    else:
        nmt=train(args.src_lang,args.trg_lang,args.segment,args.log_dir,args.batch_size,args.n_workers,args.epochs)
        result=nmt()

    log_f.close()
                        #print(src_lang+" "+trg_lang+" "+segment)
