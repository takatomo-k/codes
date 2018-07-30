import hyperparams as hp
from torch import nn,optim
from tqdm import tqdm

import sys,os,argparse,torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../../')

from util.show import *
from util.new_data import get_dataset,DataLoader, voc_cfn, get_param_size,save_wav
from util.evaluations import wer
from new_models.Tacotron.Taco_network import Vocoder
from new_models.Tacotron.Taco_network import Tacotron
import glob,io
parser = argparse.ArgumentParser()
parser.add_argument('--src_lang', type=str, help='src language')
parser.add_argument('--segment', type=str, help='src language',default="char")
parser.add_argument('--trg_lang', type=str, help='src language', default=None)
parser.add_argument('--restore_epoch', type=int, help='src language', default=None)
parser.add_argument('--log_dir', type=str, help='src language', default="./exp")
parser.add_argument('--batch_size', type=int, help='Batch size', default=16)
parser.add_argument('--n_workers', type=int, help='Batch size', default=8)
args = parser.parse_args()

class train(object):
    def __init__(self):
        self.dataset,self.path,self.collate_fn=get_dataset(args,"Tacotron")
        self.v=Vocoder().cuda()
        self.t=Tacotron(len(self.dataset.src_id2word)).cuda()
        try:
            self.v.load_state_dict(torch.load("./Vocoder"))
            self.t.load_state_dict(torch.load("./Tacotron"))
            pass
        except Exception as e:
            print(e)

    def __call__(self):
        with io.open(sys.stdin.fileno(),'r',encoding='utf-8') as sin:  # オリジナルはlatin-1
            print("please input sentence")
            for line in sin:
                input=line.strip()
                import pdb; pdb.set_trace()
                input=self.dataset.w2i(" ".join(list(input)),args.src_lang)
                mel_out,_,_=self.t(torch.from_numpy(input).type(torch.cuda.LongTensor).unsqueeze(0))
                linear_out=self.v(mel_out)

                wav = inv_spectrogram(linear_output[0].data.cpu().numpy())
                out = io.BytesIO()
                save_wav(wav, out)
                wav=out.getvalue()
                with open("./"+input_line1, 'wb') as f:
                    f.write(wav)
                    print("save wav file at  %s ..." % ("./"+input_line1+".wav"))
                print("please input sentence")

if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    taco=train()

    taco()
