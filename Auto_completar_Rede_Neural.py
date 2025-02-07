import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
torch.manual_seed(1)
context_size=2
embedding_dim=10

raw_text='''#historico de texto'''.split()

def mak_sentence(context, word_to_idx):
  idxs=[word_to_idx[i] for i in context]
  return torch.tensor(idxs, dtype=torch.long)
vocab=set(raw_text)
vocab_size=len(vocab)
word_to_idx={word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
data=[]
for i in range(2, len(raw_text)-2):
  context=[raw_text[i-2], raw_text[i-1], raw_text[i+1], raw_text[i+2]]
  target=raw_text[i]
  data.append((context, target))
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings=nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, 128)
        self.output = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = F.relu(self.proj(embeds))
        out = self.output(out)
        nll_prob = F.log_softmax(out, dim=-1)
        return nll_prob

model=CBOW(vocab_size, embedding_dim)
optimizer = optim.SGD(model.parameters(), lr=0.001)

lossers=[]
loss_function=nn.NLLLoss()
for epoch in range(100):
  for context, target in data:
    context_idx=mak_sentence(context, word_to_idx)
    model.zero_grad()
    llm_prob=model(context_idx)
    loss=loss_function(llm_prob, Variable(torch.tensor([word_to_idx[target]])))
    loss.backward()
    optimizer.step()
  lossers.append(loss.data)

context=['#qualquer palavra']
context_idx=mak_sentence(context, word_to_idx)
a=model(context_idx).data.numpy()
make_in=np.argmax(a)
print(idx_to_word[make_in]) #qual palavra vai completar
