import hyperparams as hp
from torch import nn,optim
from tqdm import tqdm
import sys,os,argparse,torch,io
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../../')
from models.Tacotron.network import Tacotron,Vocoder
import base

class train(base.base):
    def __init__(self,src_lang,trg_lang,segment,log_dir,batch_size,n_workers,epochs):
        self.name="Tacotron"
        self.data_type="TextSpeech"
        super().__init__(src_lang,trg_lang,segment,log_dir,batch_size,n_workers,epochs)

        self.model=Tacotron(len(self.dataset.src_vocab[self.segment]))
        self.Vocoder=None
        self.path=os.path.join(self.path,self.model.get_path(),segment)

        if not os.path.exists(os.path.join(self.path,"CHAMP")):
            os.makedirs(os.path.join(self.path,"CHAMP"))

        self.criterion=nn.SmoothL1Loss()
        self.stop_criterion=nn.BCEWithLogitsLoss()

        self.lr=hp.lr
        self.optimizer=optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_loss=list()
        self.dev_loss=list()
        self.load()

    def load(self,):
        return 0
        #import pdb; pdb.set_trace()
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
            print("train loss %3f dev loss %3f"%(self.train_loss[-1],min(self.dev_loss)))

    def save(self,):
        #super().__save__()
        with open(self.path+"/train_loss","w") as f:
            for i in self.train_loss:
                f.write(str(float(i))+'\n')

        with open(self.path+"/dev_loss","w") as f:
            for i in self.dev_loss:
                f.write(str(float(i))+'\n')

        if min(self.dev_loss)==self.dev_loss[-1]:
            torch.save(self.model.state_dict(),os.path.join(self.path,"CHAMP",self.name))
            print("Saving model at ",self.epoch)
            if self.dev_loss[-1] <= hp.stop_learning:
                self.epoch=self.epochs

        show_loss(self.train_loss,self.path+"/train_loss.pdf",self.criterion._get_name())
        show_loss(self.dev_loss,self.path+"/dev_loss.pdf","SmoothL1Loss")

    def eval(self,mode=None):
        src=self.data['src']['id_'+self.segment].cuda()
        trg=self.data['src']['mel'].transpose(1,2).cuda()
        txt=self.data['src']['raw_'+self.segment]

        with torch.no_grad():
            mel_out,stop_targets,att_score=self.model(src)
            length=min(mel_out.size(-1),trg.size(-1))
            loss=torch.nn.functional.l1_loss(mel_out[:,:,:length],trg[:,:,:length])

        self.total_loss+=loss.item()
        return mel_out,txt,None, "L1_loss %0.4f "%(self.total_loss/(self.i+1))

    def dump(self,hyp,ref,att_score):
        if self.Vocoder is None:
            self.Vocoder=Vocoder().cuda()
            vocoder_path=os.path.join("./exp","Vocoder",self.src_lang if self.trg_lang is None else self.trg_lang,'CHBG/CHAMP/Vocoder')
            self.Vocoder.load_state_dict(torch.load(vocoder_path))

        with torch.no_grad():
            linear_out=self.Vocoder(hyp)
        file_name=ref
        wav = inv_spectrogram(linear_out[0].cpu().numpy())
        out = io.BytesIO()
        save_wav(wav, out)
        wav=out.getvalue()
        if not os.path.exists(os.path.join(self.path,"wavs")):
            os.makedirs(os.path.join(self.path,"wavs"))

        with open(os.path.join(self.path,"wavs",file_name+".wav"),'wb') as f:
            f.write(wav)
            print("save wav file %s ..." % (file_name+".wav"))


    def train(self):
        src=self.data['src']['id_'+self.segment].cuda()
        trg=self.data['src']['mel'].transpose(1,2).cuda()
        s_trg=self.data['src']['stop_targets'].cuda()

        mel_out,stop_outputs,att_score=self.model(src,trg,self.teacher_forcing_ratio)
        length=min(mel_out.size(-1),trg.size(-1))
        mel_loss=self.criterion(mel_out[:,:,:length],trg[:,:,:length])
        stop_loss=self.stop_criterion(stop_outputs.squeeze(1)[:,:length],s_trg[:,:length])
        loss=mel_loss+stop_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

        self.optimizer.step()
        self.model.zero_grad()
        self.optimizer.zero_grad()

        self.total_loss+=(mel_loss.item()+stop_loss.item())

        return "Epoch %d tr %0.1f tts_loss %0.4f"%(self.epoch,self.teacher_forcing_ratio,self.total_loss/(self.i+1))

    def adjust_lr(self):
        if 0<= self.epoch <= 5:
            self.lr=0.001
        elif 5< self.epoch <= 10:
            self.lr=0.0005
        elif 10< self.epoch <= 15:
            self.lr=0.0003
        elif 15< self.epoch <=20:
            self.lr=0.0001
        elif 20< self.epoch <=35:
            self.lr=0.00005
        elif 35< self.epoch <=40:
            self.lr=0.00003
        elif 40< self.epoch :
            self.lr=0.00001

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def adjust_tr(self):
        if 5<= self.epoch <= 10:
            self.teacher_forcing_ratio=0.9
        elif 10< self.epoch <=20:
            self.teacher_forcing_ratio=0.7
        elif 20< self.epoch <=30:
            self.teacher_forcing_ratio=0.5
        elif 40< self.epoch <=50:
            self.teacher_forcing_ratio=0.3
        elif 50< self.epoch <=60:
            self.teacher_forcing_ratio=0.1
        else:
            self.teacher_forcing_ratio=1

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

    src_list=["japanese"]
    segment_list=["yomi_char"]
    if not os.path.exists(args.log_dir+"/Tacotron/"):
        os.makedirs(args.log_dir+"/Tacotron/")

    log_f=open(args.log_dir+"/Tacotron/log","w")
    if args.src_lang is None:
        for segment in segment_list:
            for src_lang in src_list:
                log_f.write(src_lang+"\t"+segment+"START"+"\n")
                taco=train(src_lang,None,segment,args.log_dir,args.batch_size,args.n_workers,args.epochs)
                result=taco()
                log_f.write(src_lang+"\t"+segment+result+"END"+"\n")
    else:
        taco=train(args.src_lang,None,args.segment,args.log_dir,args.batch_size,args.n_workers,args.epochs)
        result=taco()
        print(result)
