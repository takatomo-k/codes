import util.data_setting as hp
from torch.utils.data import Dataset, DataLoader
import sys,os
import librosa
import random
import numpy as np
import collections
from scipy import signal
import pickle,copy
PAD,SOS,EOS,UNK=0,1,2,3
import torch
from util.data_utils import *

DIC={'id_word','id_sub_word','id_char','id_yomi_word','id_yomi_sub','id_yomi_char'}

ROOT='/project/nakamura-lab08/Work/takatomo-k/dataset'


def get_dataset(src_lang,trg_lang,segment,log_dir,data_type,mode):
    data_dir=log_dir+'/'+mode+'/'+src_lang
    ret=CustomDataset(src_lang,trg_lang,segment,data_type,data_dir)
    return ret,data_dir

class CustomDataset(Dataset):
    '''docstring for [object Object].'''
    def __init__(self, src_lang,trg_lang,segment,data_type,data_path):
        super(Dataset, self).__init__()
        self.data_type,self.src_lang,self.trg_lang=(data_type,src_lang,trg_lang)
        self.src_data_type,self.trg_data_type= get_data_type(data_type,src_lang,trg_lang)
        print("####LOADING  DATASET####")
        self.src,self.src_vocab=load_data(os.path.join(ROOT,'BTEC','Dataset',src_lang),self.src_data_type=="speech")
        if self.trg_lang is not None:
            self.trg,self.trg_vocab=load_data(os.path.join(ROOT,'BTEC','Dataset',trg_lang),self.src_data_type=="speech")
        #import pdb; pdb.set_trace()
        if self.src_data_type=="speech":
            self.re_check(self.src)
        if self.trg_lang is not None:
            print("####PAIRWISE DATASET####")
            if self.trg_data_type=="speech":
                self.re_check(self.trg)
            self.pairwise()

        self.dump()

    def dump(self):
        print("####DATASET  CONFIGS####")
        print("SRC TRAIN Data:",len(self.src['train']['list']))
        print("SRC DEV   Data:",len(self.src['dev']['list']))
        print("SRC TEST  Data:",len(self.src['test']['list']))
        print("SRC WORD VOCAB:",len(self.src_vocab['word']))
        print("SRC SUB  VOCAB:",len(self.src_vocab['sub_word']))
        print("SRC CHAR VOCAB:",len(self.src_vocab['char']))
        print("SRC Y_WRD VOCAB:",len(self.src_vocab['yomi_word']))
        print("SRC Y_SUB  VOCAB:",len(self.src_vocab['yomi_sub']))
        print("SRC Y_CHA VOCAB:",len(self.src_vocab['yomi_char']))
        if self.trg_lang is not None:
            print("TRG TRAIN  Data:",len(self.trg['train']['list']))
            print("TRG DEV    Data:",len(self.trg['dev']['list']))
            print("TRG TEST   Data:",len(self.trg['test']['list']))
            print("TRG WORD  VOCAB:",len(self.trg_vocab['word']))
            print("TRG SUB   VOCAB:",len(self.trg_vocab['sub_word']))
            print("TRG CHAR  VOCAB:",len(self.trg_vocab['char']))
            print("TRG Y_WRD VOCAB:",len(self.trg_vocab['yomi_word']))
            print("TRG Y_SUB VOCAB:",len(self.trg_vocab['yomi_sub']))
            print("TRG Y_CHA VOCAB:",len(self.trg_vocab['yomi_char']))

    def i2w(self,seq,segment,lang):
        ret=list()
        if lang ==self.src_lang:
            vocab=self.src_vocab
        else:
            vocab=self.trg_vocab

        for s in seq:
            if s !=2:
                ret.append(vocab[segment][s])
        return ret

    def re_check(self,data):
        for mode in data.keys():
            _list=data[mode]['list']
            for label in data[mode]['list']:
                if data[mode][label]['wav'] is None:
                    import pdb; pdb.set_trace()

    def pairwise(self):
        for mode in self.src.keys():
            _list=self.src[mode]['list']
            for label in self.src[mode].keys():
                if mode =="dev":
                    if label in self.trg["train"]:
                        self.trg[mode].update({label:self.trg["train"][label]})
                if label not in self.trg[mode]:
                    _list.remove(label)
            self.src[mode]['list']=_list

    def switch_dataset(self,mode):
        self.mode=mode
        self.data_list=self.src[self.mode]['list']

    def load_wav(self,data):
        try:
            mel=np.load(data['wav'].replace('.wav','.npy').replace('wav','mel')).astype(np.float32)
            data.update({'mel':mel.transpose(1,0)})
            if self.data_type =="Speech":
                linear=np.load(data['wav'].replace('.wav','.npy').replace('wav','linear')).astype(np.float32)
                data.update({'linear':linear})
        except:
            print(data)
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        label=self.data_list[idx]
        data=dict()
        if self.src_data_type =="speech":
            self.load_wav(self.src[self.mode][label])
        data.update({"src":self.src[self.mode][label]})
        if self.trg_lang is not None:
            if self.trg_data_type =="speech":
                self.load_wav(self.trg[self.mode][label])
            data.update({'trg':self.trg[self.mode][label]})
        return data

class CustomDataset(Dataset):
    '''docstring for [object Object].'''
    def __init__(self):
        super(Dataset, self).__init__()
        self.data_type,self.src_lang,self.trg_lang=(data_type,src_lang,trg_lang)
        self.src_data_type,self.trg_data_type= get_data_type(data_type,src_lang,trg_lang)
        print("####LOADING  DATASET####")
        self.src,self.src_vocab=load_data(os.path.join(ROOT,'BTEC','Dataset',src_lang),self.src_data_type=="speech")
        if self.trg_lang is not None:
            self.trg,self.trg_vocab=load_data(os.path.join(ROOT,'BTEC','Dataset',trg_lang),self.src_data_type=="speech")
        #import pdb; pdb.set_trace()
        if self.src_data_type=="speech":
            self.re_check(self.src)
        if self.trg_lang is not None:
            print("####PAIRWISE DATASET####")
            if self.trg_data_type=="speech":
                self.re_check(self.trg)
            self.pairwise()

        self.dump()

    def dump(self):
        print("####DATASET  CONFIGS####")
        print("SRC TRAIN Data:",len(self.src['train']['list']))
        print("SRC DEV   Data:",len(self.src['dev']['list']))
        print("SRC TEST  Data:",len(self.src['test']['list']))
        print("SRC WORD VOCAB:",len(self.src_vocab['word']))
        print("SRC SUB  VOCAB:",len(self.src_vocab['sub_word']))
        print("SRC CHAR VOCAB:",len(self.src_vocab['char']))
        print("SRC Y_WRD VOCAB:",len(self.src_vocab['yomi_word']))
        print("SRC Y_SUB  VOCAB:",len(self.src_vocab['yomi_sub']))
        print("SRC Y_CHA VOCAB:",len(self.src_vocab['yomi_char']))
        if self.trg_lang is not None:
            print("TRG TRAIN  Data:",len(self.trg['train']['list']))
            print("TRG DEV    Data:",len(self.trg['dev']['list']))
            print("TRG TEST   Data:",len(self.trg['test']['list']))
            print("TRG WORD  VOCAB:",len(self.trg_vocab['word']))
            print("TRG SUB   VOCAB:",len(self.trg_vocab['sub_word']))
            print("TRG CHAR  VOCAB:",len(self.trg_vocab['char']))
            print("TRG Y_WRD VOCAB:",len(self.trg_vocab['yomi_word']))
            print("TRG Y_SUB VOCAB:",len(self.trg_vocab['yomi_sub']))
            print("TRG Y_CHA VOCAB:",len(self.trg_vocab['yomi_char']))

    def i2w(self,seq,segment,lang):
        ret=list()
        if lang ==self.src_lang:
            vocab=self.src_vocab
        else:
            vocab=self.trg_vocab

        for s in seq:
            if s !=2:
                ret.append(vocab[segment][s])
        return ret

    def re_check(self,data):
        for mode in data.keys():
            _list=data[mode]['list']
            for label in data[mode]['list']:
                if data[mode][label]['wav'] is None:
                    import pdb; pdb.set_trace()

    def pairwise(self):
        for mode in self.src.keys():
            _list=self.src[mode]['list']
            for label in self.src[mode].keys():
                if mode =="dev":
                    if label in self.trg["train"]:
                        self.trg[mode].update({label:self.trg["train"][label]})
                if label not in self.trg[mode]:
                    _list.remove(label)
            self.src[mode]['list']=_list

    def switch_dataset(self,mode):
        self.mode=mode
        self.data_list=self.src[self.mode]['list']

    def load_wav(self,data):
        try:
            mel=np.load(data['wav'].replace('.wav','.npy').replace('wav','mel')).astype(np.float32)
            data.update({'mel':mel.transpose(1,0)})
            if self.data_type =="Speech":
                linear=np.load(data['wav'].replace('.wav','.npy').replace('wav','linear')).astype(np.float32)
                data.update({'linear':linear})
        except:
            print(data)
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        label=self.data_list[idx]
        data=dict()
        if self.src_data_type =="speech":
            self.load_wav(self.src[self.mode][label])
        data.update({"src":self.src[self.mode][label]})
        if self.trg_lang is not None:
            if self.trg_data_type =="speech":
                self.load_wav(self.trg[self.mode][label])
            data.update({'trg':self.trg[self.mode][label]})
        return data




def collate_fn(batch):
    if isinstance(batch[0], collections.Mapping):
        ret=dict()
        for tag in batch[0].keys():
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

def make_stop_targets(batch_size,length, audio_lengths):
    stop_targets = torch.ones((batch_size, length)).type(torch.FloatTensor)
    for i in range(len(stop_targets)):
        stop_targets[i, 0:audio_lengths[i] - 1] *= 0
    return stop_targets
