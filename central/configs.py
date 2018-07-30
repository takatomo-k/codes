import hyperparams as hp

class Configs():
    def __init__(self,args,task):
        print("##INIT ",self.__class__.__name__)
        self.args=args
        self.task=task
        self.load_lang_settings()
        self.load_default_settings()

    def load_default_settings(self):
        self.total_epochs=self.args.epochs
        self.batch_size=self.args.batch_size

    def load_lang_settings(self):
        if self.task in {'ASR','TTS'}:
            self.src_lang=self.args.src_lang
            self.src_segment =self.args.segment
            self.trg_lang=self.src_lang
            self.trg_segment =self.args.segment
        else:
            self.src_lang=self.args.src_lang
            self.src_segment =self.args.segment
            self.trg_lang=self.args.trg_lang
            self.trg_segment =self.args.segment

    def load_network_settings(self,src_vocabsize,trg_vocabsize):
        if self.task=="ASR":
            self.in_size=80
            self.embed_drop=0.3
            self.enc_in= 128
            self.enc_hidden= 256
            self.enc_out= 256
            self.enc_drop= 0.5
            self.enc_rnn='LSTM'
            self.att_type='dot'
            self.enc_bid=True
            self.dec_in= 256
            self.dec_hidden= 256
            self.out_size= src_vocabsize
            self.dec_drop= 0.5 if self.src_segment != "char" else 0.3
            return self.in_size,self.emg_drop,self.enc_in,self.enc_hidden,self.enc_drop,self.enc_bid,self.enc_rnn,self.att_type,self.dec_in,self.dec_hidden,self.dec_drop,self.out_size

        elif self.task=="NMT":
            self.in_size= data.src_vocabsize
            self.enc_in= 128
            self.enc_hidden= 256
            self.enc_out= 256
            self.enc_drop= 0.5
            self.dec_in= 256
            self.dec_hidden= 256
            self.dec_out= data.trg_vocabsize
            self.dec_drop= 0.5
        elif self.task=="TTS":
            self.in_size= data.src_vocabsize
            self.enc_in= 128
            self.enc_hidden=256
            self.enc_out=256
            self.enc_drop=0.5
            self.dec_in=256
            self.dec_hidden=256
            self.dec_out=hp.num_mels
            self.dec_drop=0.5
