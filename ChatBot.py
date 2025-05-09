import torch, torch.nn as nn, torch.optim as optim, nltk, time, math
from nltk.tokenize import word_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

nltk.download('punkt')
def google_search(query):
  url = f"https://html.duckduckgo.com/html/?q={query.replace('?', ' ')}"
  options = Options()
  options.add_argument("--headless")
  driver = webdriver.Chrome(options=options)
  try:
    driver.get(url)
    soup=BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()
    links=[a.get("href") for a in soup.find_all("a", "result__a")]
    return links
  except:
    return []

def extrair_texto(texto):
  links=google_search(texto)
  if not links:
    return "Não consegui encontrar informações sobre isso."
  options=Options()
  options.add_argument("--headless")
  driver=webdriver.Chrome(options=options)
  texto_total=""
  for link in links:
    try:
      driver.get(f'https:{link}')
      time.sleep(2)
      soup=BeautifulSoup(driver.page_source, "html.parser")
      paragrafos=soup.find_all('p')
      texto_total+=' '.join([p.get_text() for p in paragrafos if len(p.get_text()) > 30]) + '\n'
    except Exception as e:
      print(f'Erro ao acessar {link}: {e}')
  driver.quit()
  parser=PlaintextParser.from_string(texto_total, Tokenizer('portuguese'))
  summarizer=TextRankSummarizer()
  resumo=summarizer(parser.document, 3)
  return ' '.join(str(sentenca) for sentenca in resumo)

data={"intents": [{"pattern": "Oi", "response": "Sou o PedroBot um assistente virtual"},{"pattern": "Qual a suas diretrizes", "response": "As 3 leis da robótica de Asimov"}]}

def atualizar_texto(data):
  all_words=[]
  for intent in data["intents"]:
    tokens=word_tokenize(intent["pattern"].lower())
    all_words.extend(tokens)
  all_words=sorted(set(all_words))
  word2idx={w: i for i, w in enumerate(all_words)}
  return all_words, word2idx

def tokenize_indices(sentence, word2idx):
  tokens=word_tokenize(sentence.lower())
  idxs=[word2idx[w] for w in tokens if w in word2idx]
  return torch.tensor(idxs, dtype=torch.long).unsqueeze(0)

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len):
    super().__init__()
    pe=torch.zeros(max_len, d_model)
    pos=torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term=torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2]=torch.sin(pos * div_term)
    pe[:, 1::2]=torch.cos(pos * div_term)
    pe=pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x=x+self.pe[:, :x.size(1), :]
    return x

class TransformerChatbot(nn.Module):
  def __init__(self, vocab_size, d_model, nhead, num_layers, hidden_dim, output_size, max_len):
    super().__init__()
    self.embedding=nn.Embedding(vocab_size, d_model)
    self.pos_encoder=PositionalEncoding(d_model, max_len)
    encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True)
    self.transformer_encoder=nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.classifier=nn.Linear(d_model, output_size)

  def forward(self, x):
    embeds=self.embedding(x)
    x=self.pos_encoder(embeds)
    x=self.transformer_encoder(x)
    pooled=x.mean(dim=1)
    return self.classifier(pooled)

def train_model(model, data, word2idx, epochs=50):
  criterion=nn.CrossEntropyLoss()
  optimizer=optim.Adam(model.parameters(), lr=0.01)
  for epoch in range(epochs):
    for i, intent in enumerate(data["intents"]):
      x_train=tokenize_indices(intent["pattern"], word2idx)
      y_train=torch.tensor([i], dtype=torch.long)
      output=model(x_train)
      loss=criterion(output, y_train)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

all_words, word2idx=atualizar_texto(data)
model=TransformerChatbot(vocab_size=len(word2idx),d_model=32,nhead=4,num_layers=2,hidden_dim=64,output_size=len(data['intents']),max_len=50)
train_model(model, data, word2idx, epochs=50)
def chatbot_response(text):
  x=tokenize_indices(text, word2idx)
  output=model(x)
  if x.numel()==0:
    return extrair_texto(text)
  _, predicted=torch.max(output, dim=1)
  return data["intents"][predicted.item()]["response"]

while True:
  user_input=input("Você: ")
  if user_input.lower() == "sair":
    break
  resposta=chatbot_response(user_input)
  print(f"ChatBot: {resposta}")
  data['intents'].append({'pattern': user_input, 'response': resposta})
  all_words, word2idx=atualizar_texto(data)
  model=TransformerChatbot(vocab_size=len(word2idx),d_model=32,nhead=4,num_layers=2,hidden_dim=64,output_size=len(data['intents']),max_len=50)
  train_model(model, data, word2idx, epochs=50)
