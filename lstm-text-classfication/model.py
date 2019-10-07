import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import torch.nn.functional as F

import const


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__() #确保父类被正确的初始化
        self.eps = eps  
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input):
        mu = torch.mean(input, dim=-1, keepdim=True)  #平均值
        sigma = torch.std(input, dim=-1, keepdim=True).clamp(min=self.eps)  #标准差 将元素限制在[eps,max]范围内
        output = (input - mu) / sigma
        return output * self.weight.expand_as(output) + self.bias.expand_as(output)
        #weight bias扩充到output的维度

class LSTM_Text(nn.Module):
    def __init__(self, args):
        super().__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.num_directions = 2 if self.bidirectional else 1 #是否双向

        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim,
                                         padding_idx=const.PAD)                                 
        #https://pytorch.org/docs/stable/nn.html#embedding
        #存储word embeddings并使用索引检索 key:词的索引 value：词向量
        #padding_idx 对padding_idx所在的行进行填0。padding:在将不等长的句子组成一个batch时，对那些空缺的位置补0，以形成一个统一的矩阵。                                          
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.lstm_layers,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
        #https://pytorch.org/docs/stable/nn.html#lstm
        #input个数，hidden layer个数，dropout，层数
        #dropout如果非零，在除最后一层外的每一LSTM层的输出上引入一个Dropout层（缓解过拟合）
        self.ln = LayerNorm(self.hidden_size * self.num_directions) #归一化
        self.logistic = nn.Linear(self.hidden_size * self.num_directions,
                                  self.label_size)
        #https://pytorch.org/docs/stable/nn.html#linear
        #线性变换 input,output y=xA+b

        self._init_weights()

    def _init_weights(self, scope=1.):
        self.lookup_table.weight.data.uniform_(-scope, scope)
        self.logistic.weight.data.uniform_(-scope, scope)
        self.logistic.bias.data.fill_(0)

    def init_hidden(self):
        num_layers = self.lstm_layers * self.num_directions

        weight = next(self.parameters()).data
        #通过调用迭代器的next()方法从迭代器中检索下一项
        return (Variable(weight.new(num_layers, self.batch_size, self.hidden_size).zero_()), Variable(weight.new(num_layers, self.batch_size, self.hidden_size).zero_()))

    def forward(self, input, hidden):
        encode = self.lookup_table(input)
        lstm_out, hidden = self.lstm(encode.transpose(0, 1), hidden)
        output = self.ln(lstm_out)[-1]
        return F.log_softmax(self.logistic(output)), hidden
