MAX_LEN,MIN_LEN=(100,0)
MAX_DUR,MIN_DUR=(800,1)
MIN_NUM_DATA=100
from random import shuffle
from torch.utils.data import Dataset
import os,sys,copy,pickle
import numpy as np
class CustomDataset(Dataset):
    """docstring for [object Object]."""
    def __init__(self, config):
        super(CustomDataset, self).__init__()
        print("##INIT ",self.__class__.__name__)
        self.c=config
        try:
            self.load()
        except:
            self.src,self.trg,self.data_list,self.src_vocab,self.trg_vocab=self.get_dataset()
            self.save()

    def log(self):
        log="TRAIN DATA NUM:"+str(len(self.data_list['train']))+"\n"
        log+="DEV DATA NUM:"+str(len(self.data_list['dev']))+"\n"
        log+="TEST DATA NUM:"+str(len(self.data_list['test']))+"\n"
        self.c.set_log_dataset(log)

    def __len__(self):
        return len(self.data_list[self.c.get_current_state()])

    def __getitem__(self, idx):
        label=self.data_list[self.c.current_state][idx]
        return self.extract_feature([self.src[label],self.src[label],self.trg[label],self.trg[label]])
    def save(self):
        self.c.save_dataset(self.src,self.trg,self.data_list,self.src_vocab,self.trg_vocab)
    def load(self):
        self.src,self.trg,self.data_list,self.src_vocab,self.trg_vocab=self.c.load_dataset()

    def extract_feature(self,data_list):
        ret=dict()
        src_in_key,src_out_key,trg_in_key,trg_out_key=[self.get_key()]
        #load src input feature
        for i,data,key in enumurate(zip(data_list,key_list)):
            if key is not None:
                tag:'src' if i<=2 else 'trg'
                if key in {'mel','linear'}:
                    ret.update({tag:{key:np.load(data[key])}})
                else:
                    ret.update({tag:{'id_'+key:data['id_'+key]}})
                    if self.c.is_train():
                        ret.update({tag:{'raw_'+key:data['raw_'+key]}})
        return ret

    def get_datalist(self,src_train,trg_train,src_test):
        src_key,_,_,trg_key=self.get_key()
        src_key = 'dur' if src_key in {'mel','linear'} else 'length'
        trg_key = 'dur' if trg_key in {'mel','linear'} else 'length'

        _list=[label for label in src_train.keys()]
        shuffle(_list)
        train_list=_list[0:-1000]
        dev_list=_list[-1000:len(_list)]
        #sort
        train_list=sorted(train_list, key=lambda x:(trg_train[x][trg_key],src_train[x][src_key]),reverse=True)

        test_list=[label for label in src_test.keys()]

        return {'train':train_list,'dev':dev_list,'test':test_list}

    def get_key(self):
        text_segment,task=self.c.get_segment(),self.c.get_task()
        if task  =="ASR":
            return 'mel',None,None,text_segment
        elif task=="TTS":
            return text,None,None,'mel'
        elif task=="NMT" :
            return text,None,None,text_segment
        elif task=="Speech2Text":
            return 'mel',text_segment,None,text_segment
        elif task=="Speech2Speech":
            return 'mel',text_segment,text_segment,'mel'
        elif task=="Text2Speech":
            return text_segment,text_segment,None,'mel'
        elif task=="Speech":
            return 'mel',None,None,'linear'
        else:
            assert True ,"Unknown task"

    def get_dataset(self):
        src,trg=dict(),dict()
        src_train,trg_train,src_test,trg_test,src_vocab,trg_vocab=self.c.get_dataset()
        src_train,trg_train=self.clean(src_train,trg_train)
        src_test,trg_test=self.clean(src_test,trg_test)
        data_list=self.get_datalist(src_train,trg_train,src_test)
        for label in src_train.keys():
            src.update({label:src_train[label]})
            trg.update({label:trg_train[label]})
        for label in src_test.keys():
            src.update({label:src_test[label]})
            trg.update({label:trg_test[label]})

        return src,trg,data_list,src_vocab,trg_vocab

    def clean(self,src,trg):
        labels=[label for label in src.keys()]
        for label in labels:
            if not MIN_LEN<=src[label]['length'] <=MAX_LEN:
                del src[label]

        labels=[label for label in trg.keys()]
        for label in labels:
            if not MIN_LEN<=trg[label]['length'] <=MAX_LEN:
                del trg[label]

        assert len(src)>MIN_NUM_DATA,"SRC_DATA size not enough"
        assert len(trg)>MIN_NUM_DATA,"TRG_DATA size not enough"

        if src !=trg:
            labels=[label for label in src.keys()]
            for label in labels:
                if not label in trg:
                    del src[label]

            labels=[label for label in trg.keys()]
            for label in labels:
                if not label in src:
                    del trg[label]
        assert len(src)==len(trg),"PAIR:SRC and TRG num of data not equal"
        assert len(src)>MIN_NUM_DATA,"PAIR:DATA size not enough"

        uniq=dict()
        labels=[label for label in src.keys()]
        for label in labels:
            key=src[label]['mel']+trg[label]['mel']
            if key in uniq:
                try:
                    del src[label]
                    del trg[label]
                except:
                    pass
            else:
                uniq.update({key:0})
        assert len(src)>MIN_NUM_DATA,"UNIQ:DATA size not enough"
        assert len(src)==len(trg),"UNIQ:SRC and TRG num of data not equal"


        return src,trg


def collate_fn(batch):
    if isinstance(batch[0], collections.Mapping):
        ret=dict()
        for tag in batch[0]['src'].keys():
            data=dict()
            for key in batch[0][tag].keys():
                if key in DIC:
                    text=_prepare_data([d[tag][key] for d in batch])
                    data.update({key:torch.from_numpy(text).type(torch.LongTensor)})
                elif key in {'mel','linear'}:
                    feat= [d[tag][key] for d in batch]
                    data.update({key:torch.from_numpy(_prepare_data(feat)).type(torch.FloatTensor)})
                    if key =='mel':
                        audio_lengths = [len(x) for x in feat]
                        stop_targets=make_stop_targets(len(batch),max(audio_lengths),audio_lengths)
                        data.update({'stop_targets':stop_targets})
                elif key in{'duration','length','label','wav'}:
                    pass
                else:
                    data.update({key:[d[tag][key] for d in batch]})
            ret.update({tag:data})
        return ret
    raise TypeError(('batch must contain tensors, numbers, dicts or lists; found {}'
                     .format(type(batch[0]))))
