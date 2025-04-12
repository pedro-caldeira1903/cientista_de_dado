import torch, torch.nn as nn, torch.optim as optim, nltk, numpy as np, time
from nltk.tokenize import word_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

nltk.download('punkt_tab')
def google_search(query):
  url=f"https://html.duckduckgo.com/html/?q={query.replace('?', ' ')}"
  options=Options()
  options.add_argument("--headless")
  pesquisa=webdriver.Chrome(options=options)
  try:
    pesquisa.get(url)
    soup=BeautifulSoup(pesquisa.page_source, "html.parser")
    pesquisa.quit()
    links=[a.get("href") for a in soup.find_all("a", "result__a")]
    return links[:3]
  except:
    return "Não consegui encontrar informações sobre isso."

def extrair_texto(texto):
  links=google_search(texto)
  options=Options()
  options.add_argument("--headless")
  pesquisa=webdriver.Chrome(options=options)
  texto_total=""
  for link in links:
    try:
      pesquisa.get(f'https:{link}')
      time.sleep(2)
      soup=BeautifulSoup(pesquisa.page_source, "html.parser")
      paragrafos=soup.find_all('p')
      texto_total+=' '.join([p.get_text() for p in paragrafos if len(p.get_text()) > 30]) + '\n'
    except Exception as e:
      print(f'Erro ao acessar {link}: {e}')
  pesquisa.quit()
  parser=PlaintextParser(texto_total, Tokenizer('portuguese'))
  summarizer=TextRankSummarizer()
  resumo=summarizer(parser.document, 3)
  return ' '.join(str(sentenca) for sentenca in resumo)

data={"intents": [{"pattern": "Oi", "response": "Sou o PedroBot"},{"pattern": "Qual a suas diretrizes?", "response": "As 3 leis da robotica de Asimov"}]}
all_sentece=[]
for intent in data["intents"]:
    words = word_tokenize(intent["pattern"].lower())
    all_sentece.extend(words)

all_words=sorted(set(all_sentece))
vocab=len(all_words)
def bag_of_words(sentence):
  sentence_words=word_tokenize(sentence.lower())
  bag=np.zeros(vocab, dtype=np.float32)
  for word in sentence_words:
    if word in all_words:
      bag[all_words.index(word)]=1.0
  return torch.tensor(bag).unsqueeze(0)
class Chatbot(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Chatbot, self).__init__()
    self.f1=nn.Linear(input_size, hidden_size)
    self.f2=nn.ReLU()
    self.f3=nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x=self.f1(x)
    x=self.f2(x)
    x=self.f3(x)
    return x
input_size=vocab
hidden_size=8
output_size=len(data["intents"])
modelo=Chatbot(input_size, hidden_size, output_size)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(modelo.parameters(), lr=0.01)
for epoch in range(100):
  for intent in data["intents"]:
    x_treino=bag_of_words(intent["pattern"])
    y_treino=torch.tensor([data["intents"].index(intent)], dtype=torch.long)
    saida=modelo(x_treino)
    loss=criterion(saida, y_treino)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("Treinamento concluído!")

def chatbot_response(text):
  x_test=bag_of_words(text)
  output=modelo(x_test)
  _, predicted=torch.max(output, dim=1)
  if torch.max(output).item()< 1.0:
    return extrair_texto(text)
  return data["intents"][predicted.item()]["response"]

while True:
  text=input("Você: ")
  if text.lower() == "sair":
    break
  print("ChatBot:", chatbot_response(text))
