import sys,os
import numpy as np
import pickle,copy
from tqdm import tqdm
PAD,SOS,EOS,UNK=0,1,2,3

(MIN_LEN,MAX_LEN)=(0,100)
(MIN_DUR,MAX_DUR)=(10,400)
def load_data(path,speech):
    #try:
    with open(path+"/train_set", 'rb') as f:
        train=clean_data(copy.deepcopy(pickle.load(f)),speech)
    with open(path+"/dev_set", 'rb') as f:
        dev=clean_data(copy.deepcopy(pickle.load(f)),speech)
    with open(path+"/test_set", 'rb') as f:
        test=clean_data(copy.deepcopy(pickle.load(f)),speech)
    with open(path+"/vocab", 'rb') as f:
        vocab=copy.deepcopy(pickle.load(f))
#    except Exception as e:
#        sys.exit('data loading error'+path)

    return {'train':train,'dev':dev,'test':test},vocab

def get_data_type(data_type,src_lang,trg_lang):
    if data_type=="TextText":
        return 'text','text'
    elif data_type=="TextSpeech" and not trg_lang is None:
        return 'text','speech'
    elif data_type=="SpeechText" and not trg_lang is None:
        return 'speech','text'
    else:
        return 'speech','speech'

def clean_data(data,speech):
    ret=dict()
    _list=list()
    frg=True
    wav_none=0
    len_false=0
    dur_false=0
    for label in data.keys():
        if label=='list':
            continue
        if speech and data[label]['wav'] is None:
            wav_none+=1
            continue
        if speech and not (MIN_DUR<data[label]['duration']<MAX_DUR):
            dur_false
            continue
        if 'length' in data[label] and not (MIN_DUR<data[label]['length']<MAX_DUR):
            len_false+=1
            continue
        if (not speech) and data[label]['wav'] is not None:
            data[label]['wav']=None

        _list.append(label)
        ret.update({label:data[label]})
    ret.update({'list':_list})
    print("WAVDEL:%d LENDEL:%d DURDEL%d"%(wav_none,len_false,dur_false))
    return ret
