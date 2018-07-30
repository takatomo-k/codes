import hyperparams as hp
from torch import nn,optim
from tqdm import tqdm
import sys,os,argparse,torch,shutil
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../../')

from central.base import Base
from util.evaluations import wer
from models.ASR.network import ASR
import base
class train(Base):
    def __init__(self,args):
        super().__init__(args,"ASR")
        self.path=os.path.join(self.path,self.model.get_path(),self.segment)
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
            print("Epoch %d train loss %3f dev loss %4f"%(self.epoch,self.train_loss[-1],self.dev_loss[-1]))

    def save(self,):
        #super().__save__()
        import pdb; pdb.set_trace()
        with open(self.path+"/train_loss","w") as f:
            for i in self.train_loss:
                f.write(str(float(i))+'\n')

        with open(self.path+"/dev_loss","w") as f:
            for i in self.dev_loss:
                f.write(str(float(i))+'\n')

        if min(self.dev_loss)==self.dev_loss[-1]:
            torch.save(self.model.state_dict(),os.path.join(self.path,"CHAMP",self.name))
            print("Saving model at ",self.epoch)
            shutil.copy(self.log_path,self.log_path.replace("Latest","Best"))
            if self.dev_loss[-1] <= hp.stop_learning:
                self.epoch=self.epochs

        #show_loss(self.train_loss,os.path.join(self.path,"train_loss.pdf"),self.criterion._get_name())
        #show_loss(self.dev_loss,os.path.join(self.path,"dev_loss.pdf"),"WER")

    def eval(self):
#        import pdb; pdb.set_trace()
        src=self.data['src']['mel'].cuda()
        ref=self.data['src']['raw_'+self.segment][0]

        with torch.no_grad():
            out_txt,att_score=self.model(src)

        hyp=self.dataset.i2w(out_txt[0].argmax(-1).cpu().numpy(),self.segment,self.src_lang)
        loss=wer(hyp,ref)
        self.total_loss+=loss
        return hyp,ref,att_score[0].cpu().numpy(), "WER %0.1f "%(self.total_loss/(self.i+1))

    def dump(self,hyp,ref,att_score):
        f_name="_".join(ref)
        self.log_path=os.path.join(self.path,"Attention",self.mode,"Latest")
        show_attention(att_score,f_name,self.log_path,label_x=None,label_y=hyp)

    def train(self):
        src,trg=self.data['src']['mel'].cuda(),self.data['src']['id_'+self.segment].cuda()
        out_txt,att_score=self.model(src,trg,self.teacher_forcing_ratio)
        loss=self.criterion(out_txt.transpose(1,2),trg)
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optimizer.step()
        self.model.zero_grad()
        self.optimizer.zero_grad()
        self.total_loss+=loss.item()

        return "Epoch %d tr %0.1f CE_loss %0.1f"%(self.epoch,self.teacher_forcing_ratio,self.total_loss/(self.i+1))


if __name__ == '__main__':
    #import pdb; pdb.set_trace()
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

    src_list=["english"]
    segment_list=["char","word","sub_word"]
    segment_list_ja=["char","word","sub_word","yomi_char","yomi_sub","yomi_word"]

    if not os.path.exists(args.log_dir+"/ASR/"):
        os.makedirs(args.log_dir+"/ASR/")

    log_f=open(args.log_dir+"/ASR/log","w")
    if args.src_lang is None:
            for src_lang in src_list:
                if src_lang =="japanese":
                    for segment in segment_list_ja:
                        log_f.write(src_lang+"\t"+segment+"START"+"\n")
                        asr=train(args)
                        result=asr()
                        log_f.write(src_lang+"\t"+segment+result+"END"+"\n")
                else:
                    for segment in segment_list:
                        log_f.write(src_lang+"\t"+segment+"START"+"\n")
                        asr=train(args)
                        result=asr()
                        log_f.write(src_lang+"\t"+segment+result+"END"+"\n")
            log_f.close()
    else:
        asr=train(args)
        result=asr()
