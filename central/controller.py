from pathlib import Path
import os,sys,copy,pickle,torch
from datetime import datetime
from models.ASR.network import ASR

class Controller():
    def __init__(self,config):
        print("##INIT ",self.__class__.__name__)

        self.c=config
        self.current_state='train'
        self.current_epoch=0
        self.log_dir=os.path.join("./exp_new",self.c.task,self.c.src_lang)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def init_model(self):
        import pdb; pdb.set_trace()
        src_vocabsize=len(self.src_vocab[self.c.src_segment])
        trg_vocabsize=len(self.trg_vocab[self.c.trg_segment])
        if self.c.task=='ASR':
            model_path=os.path.join("./exp_new",self.c.task,self.c.src_lang,self.c.src_segment,'Latest','ASR')
            model=ASR(load_network_settings(src_vocabsize,trg_vocabsize))
        try:
            model.load_state_dict(torch.load(os.path.join(self.model_path)))
        except Exception as e:
            pass
        return model

    def load_results(self,model):
        pass
    def is_train(self):
        return self.current_state=='train'
    def is_dev(self):
        return self.current_state=='dev'
    def is_test(self):
        return self.current_state=='test'

    def is_end(self):
        return self.current_epoch>=self.c.total_epochs

    def init_logs(self):
        log=datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.set_log_train(log)
        self.set_log_dev(log)
        self.set_log_test(log)
        self.set_log_dataset(log)
        self.set_log_model(log)

    def set_log_dataset(self,log):
        with open(os.path.join(self.log_dir,'data.log'),'a') as f:
            f.write(log)
    def set_log_model(self,):
        with open(os.path.join(self.log_dir,'model.log'),'a') as f:
            f.write(log)
    def set_log_train(self):
        with open(os.path.join(self.log_dir,'train.log'),'a') as f:
            f.write(log)

    def set_log_dev(self):
        with open(os.path.join(self.log_dir,'develop.log'),'a') as f:
            f.write(log)

    def set_log_test(self):
        with open(os.path.join(self.log_dir,'test.log'),'a') as f:
            f.write(log)

    def set_current_state(self,state):
        self.current_state=state
    def get_task(self):
        return self.c.task
    def get_src_lang(self):
        return self.c.src_lang
    def get_segment(self):
        return self.c.src_segment
    def get_trg_lang(self):
        return self.c.src_lang
    def get_task(self):
        return self.c.task
    def get_current_state(self):
        return self.current_state

    def load_pickle(self,path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_pickle(self,path,data):
        with open(path,'wb')as f:
            pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)

    def get_dataset(self):
        src_train=self.load_pickle(os.path.join('BTEC','train',self.c.src_lang,'data'))
        src_vocab=self.load_pickle(os.path.join('BTEC','train',self.c.src_lang,'vocab'))
        src_test =self.load_pickle(os.path.join('BTEC','test',self.c.src_lang,'data'))
        if self.c.task in {"ASR","TTS"}:
            trg_train=src_train
            trg_test =src_test
            trg_vocab=src_vocab
        else:
            trg_train=self.load_pickle(os.path.join('BTEC','train',self.c.trg_lang,'data'))
            trg_vocab=self.load_pickle(os.path.join('BTEC','train',self.c.trg_lang,'vocab'))
            trg_test =self.load_pickle(os.path.join('BTEC','test',self.c.trg_lang,'data'))

        return src_train,trg_train,src_test,trg_test,src_vocab,trg_vocab

    def save_dataset(self,src,trg,data_list,src_vocab,trg_vocab):
        self.save_pickle(os.path.join(self.log_dir,'SRC_DATA'),src)
        self.save_pickle(os.path.join(self.log_dir,'DATA_LIST'),data_list)
        self.save_pickle(os.path.join(self.log_dir,'SRC_VOCAB'),src_vocab)
        if self.c.task not in {"ASR","TTS"}:
            self.save_pickle(os.path.join(self.log_dir,'TRG_DATA'),trg)
            self.save_pickle(os.path.join(self.log_dir,'TRG_VOCAB'),trg_vocab)
        print("Save dataset:",self.log_dir)
        self.src_vocab,self.trg_vocab=src_vocab,trg_vocab

    def load_dataset(self):
        src=self.load_pickle(os.path.join(self.log_dir,'SRC_DATA'))
        src_vocab=self.load_pickle(os.path.join(self.log_dir,'SRC_VOCAB'))
        data_list=self.load_pickle(os.path.join(self.log_dir,'DATA_LIST'))

        if self.c.task in {"ASR","TTS"}:
            trg=src
            trg_vocab=src_vocab
        else:
            trg=self.load_pickle(os.path.join(self.log_dir,'TRG_DATA'))
            trg_vocab=self.load_pickle(os.path.join(self.log_dir,'TRG_VOCAB'))

        print("Load dataset:",self.log_dir)
        self.src_vocab,self.trg_vocab=src_vocab,trg_vocab
        return src,trg,data_list,src_vocab,trg_vocab
