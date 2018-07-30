import torch
from torch import nn


class MLP(nn.Module):
    """docstring for [object Object]."""
    def __init__(self, insize,hidden_size,drop):
        super(MLP, self).__init__()
        self.mlp =nn.Sequential()
        for i,d in enumerate(drop):
            #import pdb; pdb.set_trace()
            self.mlp.add_module("l_"+str(i),nn.Linear(insize,hidden_size))
            self.mlp.add_module("r_"+str(i),nn.ReLU())
            self.mlp.add_module("d_"+str(i),nn.Dropout(d))


            insize=hidden_size

    def forward(self,input):
        return self.mlp(input)


class CustomEmbedding(nn.Module):
    def __init__(self,input_size,embed_size,drop):
        super(CustomEmbedding, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        #self.diging=nn.Linear(embed_size,input_size)
        self.drop=nn.Dropout(drop)


    def forward(self,text):
        if isinstance(text,torch.cuda.FloatTensor):
            if text.dim()>2:
                embed=text.bmm(self.embedding.weight.unsqueeze(0).repeat(text.size(0),1,1)).size()
            else:
                embed=text.mm(self.embedding.weight)
        else:
            embed=self.embedding(text)
        return self.drop(embed)

    def emb(self,input):
        return self.drop(input.mm(self.embedding.weight))

    def inverse(self,hidden):
        #import pdb; pdb.set_trace()
        #return self.diging(hidden)
        return hidden.mm(self.embedding.weight.transpose(0,1))

    def __compare__(self):
        pass
