import hyperparams as hp
import sys,os,argparse,torch
from torch import nn,optim
from torch.nn import functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../../')
from tqdm import tqdm

from util.show import *
from util.new_data import get_dataset,DataLoader, taco2_cfn, get_param_size
from util.evaluations import bleu
from new_models.Tacotron2.Tacotron2_network import Tacotron2

parser = argparse.ArgumentParser()
parser.add_argument('--src_lang', type=str, help='src language')
parser.add_argument('--segment', type=str, help='src language',default="word")
parser.add_argument('--trg_lang', type=str, help='src language', default=None)
parser.add_argument('--restore_epoch', type=int, help='src language', default=None)
parser.add_argument('--log_dir', type=str, help='src language', default="./exp")
parser.add_argument('--batch_size', type=int, help='Batch size', default=16)
parser.add_argument('--n_workers', type=int, help='Batch size', default=8)

args = parser.parse_args()

class train(object):
    def __init__(self):
        self.dataset,current_path=get_dataset(args,"Tacotron2")
        self.model=Tacotron2(len(self.dataset.src_id2word))
        self.path=current_path+"/"+self.model.get_path()+"/"+args.segment
        if not os.path.exists(self.path):
            os.makedirs(self.path+"/CHAMP")

        self.criterion=nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer=optim.Adam(self.model.parameters(), lr=hp.lr)
        self.teacher_forcing_ratio=1.
        self.epoch=0
        self.timesteps=0
        self.train_loss=list()
        self.dev_loss=list()

    def load(self,):
        #load train loss
        try:
            #import pdb; pdb.set_trace()
            self.model.load_state_dict(torch.load(self.path+"/Tacotron2"))
            print("Load model from:",self.path)
        except :
            print("Start new training")
            return 0

        if os.path.exists(self.path+"/train_loss"):
            with open(self.path+"/train_loss") as f:
                for i in f.readlines():
                    i=i.strip()
                    self.train_loss.append(float(i))
                    self.teacher_forcing_ratio=float(i)

        if os.path.exists(self.path+"/dev_loss"):
            with open(self.path+"/dev_loss") as f:
                for i in f.readlines():
                    i=i.strip()
                    self.dev_loss.append(float(i))

        self.epoch=len(self.dev_loss)
        self.timesteps=len(self.train_loss)*1000

    def save(self,):
        with open(self.path+"/train_loss","w") as f:
            for i in self.train_loss:
                f.write(str(float(i))+'\n')

        with open(self.path+"/dev_loss","w") as f:
            for i in self.dev_loss:
                f.write(str(float(i))+'\n')
        torch.save(self.model.state_dict(),self.path+"/Tacotron2")

        if max(self.dev_loss)==self.dev_loss[-1]:
            torch.save(self.model.state_dict(),self.path+"/CHAMP/Tacotron2")
            print("Saving model")
            if self.dev_loss[-1] >= hp.stop_learning:
                self.epoch=hp.epoch

        show_loss(self.train_loss,self.path+"/train_loss.pdf",self.criterion._get_name())
        show_loss(self.dev_loss,self.path+"/dev_loss.pdf","BLEU")

    def develop(self):
        self.dataset.develop()
        self.model.train(False)
        dataloader = DataLoader(self.dataset, batch_size=1,
        shuffle=False, collate_fn=taco2_cfn, drop_last=True, num_workers=1,pin_memory=True)
        pbar=tqdm(enumerate(dataloader),total=len(dataloader), unit=' data')
        dev_bleu=0
        for i, (src_txt,trg_txt) in pbar:
            with torch.no_grad():
                out_txt,att_score=self.model(src_txt.cuda())

            for src,ref,out,score in zip(src_txt.numpy(),trg_txt.numpy(),out_txt.cpu().numpy(),att_score.cpu().numpy()):
                src_sen=self.dataset.i2w4eval(src,args.src_lang)
                ref_sen=self.dataset.i2w4eval(ref,args.trg_lang)
                out_sen=self.dataset.i2w4eval(np.argmax(out,axis=-1),args.trg_lang)
                dev_bleu+=bleu(out_sen,ref_sen)
                if i<15:
                    src=self.dataset.i2w(src,args.src_lang)
                    out=self.dataset.i2w(np.argmax(out,axis=-1),args.trg_lang)
                    show_attention(score,ref_sen,os.path.join(self.path,"Develop","Attention",str(self.epoch)),label_x=src.split(),label_y=out.split())

            pbar.set_description("Develop BLEU:%4f "%(dev_bleu/(i+1)) )
        return dev_bleu/(len(dataloader))

    def test(self):
        self.dataset.test()
        self.model.train(False)
        dataloader = DataLoader(self.dataset, batch_size=1,
        shuffle=False, collate_fn=taco2_cfn, drop_last=True, num_workers=1,pin_memory=True)
        pbar=tqdm(enumerate(dataloader),total=len(dataloader), unit=' data')
        loss=0
        with open(self.path+"asr_result","w") as f:
            for i, (src_txt,trg_txt) in pbar:
                with torch.no_grad():
                    out_txt,att_score=self.model(src_txt.cuda())

                for src,ref,out,score in zip(src_txt.numpy(),trg_txt.numpy(),out_txt.cpu().numpy(),att_score.cpu().numpy()):
                    src_sen=self.dataset.i2w4eval(src,args.src_lang)
                    ref_sen=self.dataset.i2w4eval(ref,args.trg_lang)
                    out_sen=self.dataset.i2w4eval(np.argmax(out,axis=-1),args.trg_lang)
                    f.write(out_sen+"|"+ref_sen+"\n")
                    loss+=bleu(out_sen,ref_sen)
                    if i<15:
                        src=self.dataset.i2w4eval(src,args.src_lang)
                        out=self.dataset.i2w(np.argmax(out,axis=-1),args.trg_lang)
                        show_attention(score,ref_sen,os.path.join(self.path,"test","Attention",str(self.epoch)),label_x=src.split(),label_y=out.split())
                pbar.set_description(f'Test BLEU: {loss/(i+1):.4f}')
            f.write("BLEU:"+str(loss/(len(dataloader))))
#            pbar.set_description("DEV WER %s" % (str(round(loss/i),3)))

    def train(self):
        self.model.train(True)
        self.dataset.train()
        self.adjust_learning_rate()
        dataloader = DataLoader(self.dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=taco2_cfn, drop_last=True, num_workers=args.n_workers,pin_memory=True)
        pbar=tqdm(enumerate(dataloader),total=len(dataloader), unit=' batches')
        train_loss=0
        train_bleu=0
        for i, (src_txt,_,trg_mel,__) in pbar:

            #self.model.zero_grad()

            out_txt,att_score=self.model(src_txt.cuda(),trg_mel.cuda())
            loss=self.criterion(out_txt.transpose(1,2),trg_mel.cuda())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
            self.model.zero_grad()
            self.optimizer.zero_grad()

            train_loss+=loss.item()
            #self.teacher_forcing_ratio=train_loss/(i+1)
            if self.timesteps % 1000  == 0 and self.timesteps>1:
                self.train_loss.append(train_loss/(i+1))
            self.timesteps+=1
            if len(dataloader)-i<10:
                for src,ref,out,score in zip(src_txt.numpy(),trg_txt.numpy(),out_txt.detach().cpu().numpy(),att_score.detach().cpu().numpy()):
                    src_sen=self.dataset.i2w4eval(src,args.src_lang)
                    ref_sen=self.dataset.i2w4eval(ref,args.trg_lang)
                    out_sen=self.dataset.i2w4eval(np.argmax(out,axis=-1),args.trg_lang)
                    train_bleu+=bleu(out_sen,ref_sen)/args.batch_size

                #import pdb; pdb.set_trace()
                pbar.set_description(f'Train loss: {train_loss/(i+1):.4f} BLEU: {train_bleu/(10-len(dataloader)+i):.4f}')
            else:
                pbar.set_description(f'Train loss: {train_loss/(i+1):.4f}')



    def adjust_learning_rate(self):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if self.epoch == 2:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0005

        elif self.epoch == 5:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0003

        elif self.epoch == 10:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0001
        if self.epoch>10:
            self.teacher_forcing_ratio=self.epoch/(self.epoch-9)

    def __call__(self):
        #self.model=nn.DataParallel(self.model).cuda()
        #self.model=nn.parallel.DistributedDataParallel(self.model).cuda()
        self.model.cuda()
        while self.epoch < hp.epochs:
            self.train()
            self.dev_loss.append(self.develop())
            self.save()
            self.epoch+=1
        self.model.load_state_dict(torch.load(self.path+"/CHAMP/Tacotron2"))
        self.test()

if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    tacoron2=train()
    tacoron2.load()
    tacoron2()
