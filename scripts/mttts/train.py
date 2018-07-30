import hyperparams as hp
import sys,os,argparse,torch,io
from torch import nn,optim
from torch.nn import functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../../')
from tqdm import tqdm
from util.show import *
from util.data import inv_spectrogram,save_wav
from util.evaluations import BLEUp1
from models.MTTTS.network import MTTTS
from models.Tacotron.network import Vocoder
import base

class train(base.base):
    def __init__(self,src_lang,trg_lang,segment,log_dir,batch_size,n_workers,epochs,args):
        self.name="MTTTS"
        self.data_type="TextSpeech"
        super().__init__(src_lang,trg_lang,segment,log_dir,batch_size,n_workers,epochs)
        self.model=MTTTS(len(self.dataset.src_id2word),len(self.dataset.trg_id2word))
        self.path=os.path.join(self.path,self.model.get_path(),segment)

        if not os.path.exists(self.path):
            os.makedirs(self.path+"/CHAMP")
        self.ce_criterion=nn.CrossEntropyLoss(ignore_index=0)
        self.l1_criterion=nn.SmoothL1Loss()
        self.mse_criterion=nn.MSELoss()

        self.stop_criterion=nn.BCEWithLogitsLoss()
        self.lr=hp.lr
        self.optimizer=optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_loss=list()
        self.dev_loss=list()
        self.phase=0
        self.load()
        self.n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)

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
            self.model.load(args)
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
        show_loss(self.dev_loss,self.path+"/dev_loss.pdf","BLEUp1")

    def test(self):
        super().__test__()
        loss=0
        with open(os.path.join(self.path,"result"),"w") as f:
            for i, data in self.pbar:
                with torch.no_grad():
                    out_txt,mel_out,stop_targets,nmt_att_score,tts_att_score=self.model(data['src_txt'].cuda())
                    length=min(mel_out.size(-1),data['trg_mel'].size(-1))
                    loss+=torch.nn.functional.l1_loss(mel_out[:,:,:length].cpu(),data['trg_mel'][:,:,:length])
                    linear_out=self.voc(data['trg_mel'].cuda())
                    file_name=self.dataset.i2w(out_txt.cpu().argmax(-1).numpy()[0],trg_lang).replace(" ","")
                    #import pdb; pdb.set_trace()
                    wav = inv_spectrogram(linear_out[0].data.cpu().numpy())
                    #wav = inv_spectrogram(data['trg_linear'].numpy()[0])
                    out = io.BytesIO()
                    save_wav(wav, out)
                    wav=out.getvalue()
                    with open("./"+self.path+"/CHAMP/"+file_name+".wav", 'wb') as f:
                        f.write(wav)
                        print("save wav file at  %s ..." % ("./"+file_name+".wav"))
            self.pbar.set_description(f'Test BLEUp1: {loss/(i+1):.4f}')
            f.write("BLEUp1:"+str(loss/(len(self.dataloader))))

    def eval(self,mode=None):
        if self.total_loss==0:
            self.ce_loss=0
        src=self.data['src_txt'].cuda()
        trg=self.data['trg_mel']
        with torch.no_grad():
            out_txt,mel_out,stop_targets,nmt_att_score,tts_att_score=self.model(src)
            length=min(mel_out.size(-1),trg.size(-1))
            loss=torch.nn.functional.l1_loss(mel_out[:,:,:length].cpu(),trg[:,:,:length])
        ref=self.dataset.i2w4eval(self.data['trg_txt'][0].numpy(),self.trg_lang)
        hyp=self.dataset.i2w4eval(out_txt,self.trg_lang)

        self.ce_loss+=BLEUp1(hyp,ref)
        self.total_loss+=loss
        return loss,hyp,ref, "BLEU+1 %0.1f L1_loss %0.1f "%(self.ce_loss/(self.i+1),self.total_loss/(self.i+1))

    def train(self):
        if self.total_loss==0:
            self.ce_loss    =0

        src_txt=self.data['src_txt'].cuda()
        trg_txt=self.data['trg_txt'].cuda()
        trg_mel=self.data['trg_mel'].cuda()
        s_trg=self.data['stop_targets'].cuda()
        out_txt,mel_out,stop_targets,nmt_att_score,tts_att_score=self.model(src_txt,trg_txt,trg_mel,self.teacher_forcing_ratio)

        mel_loss=self.l1_criterion(mel_out,trg_mel)
        ce_loss=self.ce_criterion(out_txt.transpose(1,2),trg_txt)
        stop_loss=self.stop_criterion(stop_targets.squeeze(1),s_trg)
        loss=stop_loss+mel_loss+ce_loss
        loss.backward()

        self.ce_loss+=ce_loss.item()
        self.total_loss+=(stop_loss.item()+mel_loss.item())
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optimizer.step()
        self.model.zero_grad()
        self.optimizer.zero_grad()

        return "Epoch %d tr %0.1f tts_loss %0.4f CE_loss %0.1f"%(self.epoch,self.teacher_forcing_ratio,self.total_loss/(self.i+1),self.ce_loss/(self.i+1))


    def adjust_lr(self):
        if 0<= self.epoch <= 5:
            self.lr=0.0001
        elif 5< self.epoch <= 10:
            self.lr=0.00005
        elif 10< self.epoch <= 15:
            self.lr=0.00003
        elif 15< self.epoch <=20:
            self.lr=0.00001
        elif 20< self.epoch <=35:
            self.lr=0.000005
        elif 35< self.epoch <=40:
            self.lr=0.000003
        elif 40< self.epoch :
            self.lr=0.000001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def adjust_tr(self):
        if 0<= self.epoch <= 10:
            self.teacher_forcing_ratio=1.
        elif 10< self.epoch <= 20:
            self.teacher_forcing_ratio=1
        elif 20< self.epoch <=30:
            self.teacher_forcing_ratio=1
        else:
            self.teacher_forcing_ratio=1


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

    Cascade=train(args.src_lang,args.trg_lang,args.segment,args.log_dir,args.batch_size,args.n_workers,args.epochs,args)
    result=Cascade()
