import torch
from torch import nn


class BaseEncoder(nn.Module):
    """docstring for [object Object]."""
    def __init__(self, insize,hidden_size,drop,bid,rnn):
        super(BaseEncoder, self).__init__()
        self.nets=nn.ModuleList()
        self.drops=nn.ModuleList()

        for i,(d,b,r) in enumerate(zip(drop,bid,rnn)):
            self.nets.append(getattr(nn,r)(input_size=insize, hidden_size=int(hidden_size/2) if b else hidden_size,bidirectional=b,batch_first=True))
            self.drops.append(nn.Dropout(p=d))
            insize=hidden_size

    def forward(self,input,skip_step=0):
        for n,d in zip(self.nets,self.drops):
            input,hidden=n(input)
            input=d(input)
            if skip_step >1:
                input=input[:,::skip_step,:]
        return input

class StandardDecoder(nn.Module):
    def __init__(self, insize,hidden_size,out_size,drop):
        super(StandardDecoder, self).__init__()
        self.rnn_nets=nn.ModuleList()
        self.drops=nn.ModuleList()
        self.out_layer=nn.Linear(hidden_size,out_size)
        for d in drop:
            self.rnn_nets.append(nn.LSTM(insize,hidden_size,batch_first=True))
            self.drops.append(nn.Dropout(d))
            insize=hidden_size

    def init(self,memory):
        self.hidden=(memory[:,-1,:].unsqueeze(0).contiguous(),memory[:,-1,:].unsqueeze(0).contiguous())
        self.memory=memory

    def forward(self,input,att_fn):
        #import pdb; pdb.set_trace()
        input=input.unsqueeze(1)
        for rnn,drop in zip(self.rnn_nets,self.drops):
            input,(h,c)=rnn(input,self.hidden)
            input=drop(input)
            self.hidden=(drop(h),drop(c))
        return self.out_layer(input).squeeze(1),input,input
class LuongDecoder(nn.Module):
    """docstring for [object Object]."""
    def __init__(self, insize,hidden_size,out_size,drop):
        super(LuongDecoder, self).__init__()
        #self.cell=nn.GRUCell(insize,hidden_size)
        self.rnn_nets=nn.ModuleList()
        self.drops=nn.ModuleList()
        self.concat=nn.Linear(hidden_size*2,out_size)
        for d in drop:
            self.rnn_nets.append(nn.LSTM(insize,hidden_size,batch_first=True))
            self.drops.append(nn.Dropout(d))
            insize=hidden_size


    def forward(self,input,att_fn):
        #import pdb; pdb.set_trace()
        input=input.unsqueeze(1)
        for rnn,drop in zip(self.rnn_nets,self.drops):
            input,(h,c)=rnn(input,self.hidden)
            input=drop(input)
            self.hidden=(drop(h),drop(c))


        #import pdb; pdb.set_trace()
        #self.last_h=self.cell(input,self.last_h)
        #self.last_h=self.drop(self.last_h)
        #import pdb; pdb.set_trace()
        att_weights=att_fn(input,self.memory)
        context=att_weights.bmm(self.memory)
        #import pdb; pdb.set_trace()
        output=self.concat(torch.cat((input, context), -1))

        return output.squeeze(1),att_weights.squeeze(1),context.squeeze(1)

    def init(self,memory):
        self.memory=memory
        self.hidden=(memory[:,-1,:].unsqueeze(0).contiguous(),memory[:,-1,:].unsqueeze(0).contiguous())

class BahdanauDecoder(nn.Module):
    """docstring for [object Object]."""
    def __init__(self, insize,hidden_size,out_size,drop):
        super(BahdanauDecoder, self).__init__()
        self.cell=nn.GRUCell(embedding_size+hidden_size,hidden_size)
        self.concat=nn.Linear(hidden_size*2,out_size)
        self.drop=nn.Dropout(drop)
    def forward(self,input,att_fn):
        att_weights=att_fn(self.last_h,self.memory)
        context=att_weights.bmm(self.memory)
        self.last_h=self.cell(torch.cat((input,context),-1),self.last_h)

        output=self.concat(torch.cat(self.last_h,context))
        output=F.tanh(self.drop(output))

        return output.squeeze(1),att_weights.squeeze(1),context.squeeze(1)



    def init(self,memory):
        self.memory=memory
        self.last_h=memory[:,-1,:]
