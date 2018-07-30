import sys,os,argparse,torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from central.data import CustomDataset,collate_fn
from central.controller import Controller
from central.show import Visualizer
from central.configs import Configs

class Base(object):
    def __init__(self,args,task):
        import pdb; pdb.set_trace()
        print("##INIT ",self.__class__.__name__)
        self.config=Configs(args,task)
        self.controller=Controller(self.config)
        self.visualizer=Visualizer(self.controller)
        self.dataset=CustomDataset(self.controller)

        self.controller.init_model()

    def __load__(self,):
        if os.path.exists(os.path.join(self.path,self.name)):
            try:
                self.model.load_state_dict(torch.load(os.path.join(self.path,self.name)))
                print('Load model from:',self.path)
            except Exception as e:
                print(e,self.path,self.name)
                print('Start new training')
                return False
        else:
            print('Start new training')
            return False
        return True

    def __save__(self,):
        torch.save(self.model.state_dict(),os.path.join(self.path,self.name))

    def dump(self,hyp,ref,att_score,path):
        pass

    def loop(self,mode):
        self.mode=mode
        self.total_loss=0
        self.model.train(mode=='train')
        self.dataset.switch_dataset(mode)
        batch_size,num_workers=(self.batch_size,self.num_workers) if mode =='train' else (1,1)
        dataloader = DataLoader(self.dataset, batch_size=batch_size,
        shuffle= mode=='train', collate_fn=collate_fn, drop_last= mode=='train', num_workers=num_workers,pin_memory=True)
        pbar=tqdm(enumerate(dataloader),total=len(dataloader))

        for self.i, self.data in pbar:
            if mode =='train':
                log=self.train()

            else:
                hyp,ref,att_score,log=self.eval()
                self.dump(hyp,ref,att_score) if self.i < self.num_dumps else None
                #    f.write(str(loss)+'|'+hyp+'|'+ref+'\n')
            pbar.set_description(mode+' '+log)

        if mode=='train':
            self.train_loss.append(self.total_loss/len(pbar))
        elif mode=='dev':
            self.dev_loss.append(self.total_loss/len(pbar))

        #import pdb; pdb.set_trace()

        return str(self.total_loss/len(pbar))

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
        if 0<= self.epoch <= 10:
            self.teacher_forcing_ratio=1.
        elif 10< self.epoch <= 20:
            self.teacher_forcing_ratio=1
        elif 20< self.epoch <=30:
            self.teacher_forcing_ratio=1
        else:
            self.teacher_forcing_ratio=1



    def __call__(self):
        self.model.cuda()
        while self.epoch < self.epochs:
            self.adjust_lr()
            self.adjust_tr()
            self.loop('dev')
            self.__save__()
            self.loop('train')
            self.save()
            self.epoch+=1

        return self.loop('test')
