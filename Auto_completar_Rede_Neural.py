#esse código é uma rede neural linear que usa embbeding(que transforma texto em tuplas de números e relaciona as palavras para compreender melhor suas relações) que resolve qual será a proxima palavra que é mais comum de você utilizar. 
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
torch.manual_seed(1)
tamanho_contexto=2
dimensao_embeeding=10

texto='''#historico de texto'''.split()

def mak_sentence(contexto, word_to_idx):
  idxs=[word_to_idx[i] for i in contexto]
  return torch.tensor(idxs, dtype=torch.long)
vocabulario=set(texto)
vocabulario_tamanho=len(vocabulario)
word_to_idx={word: i for i, word in enumerate(vocabulario)}
idx_to_word = {i: word for i, word in enumerate(vocabulario)}
dado=[]
for i in range(2, len(texto)-2):
  contexto=[texto[i-2], texto[i-1], texto[i+1], texto[i+2]]
  alvo=texto[i]
  dado.append((contexto, alvo))
class CBOW(nn.Module):
    def __init__(self, vocabulario_tamanho, dimensao_embeeding):
        super(CBOW, self).__init__()
        self.embeddings=nn.Embedding(vocab_size, dimensao_embeeding)
        self.proj = nn.Linear(dimensao_embeeding, 128)
        self.output = nn.Linear(128, vocabulario_tamanho)

    def forward(self, entrada):
        embeds=sum(self.embeddings(inputs)).view(1, -1)
        saida=F.relu(self.proj(embeds))
        saida=self.output(saida)
        nll_prob=F.log_softmax(saida, dim=-1)
        return nll_prob

model=CBOW(vocab_size, dimensao_embeeding)
otimizador=optim.SGD(model.parameters(), lr=0.001)

lossers=[]
loss_function=nn.NLLLoss()
for epoca in range(100):
  for contexto, alvo in dado:
    contexto_idx=mak_sentence(contexto, word_to_idx)
    model.zero_grad()
    llm_prob=model(contexto_idx)
    loss=loss_function(llm_prob, Variable(torch.tensor([word_to_idx[alvo]])))
    loss.backward()
    otimizador.step()
  lossers.append(loss.data)

contexto=['#qualquer palavra']
contexto_idx=mak_sentence(contexto, word_to_idx)
a=model(contexto_idx).data.numpy()
fazer_entrada=np.argmax(a)
print(idx_to_word[fazer_entrada]) #qual palavra vai completar
