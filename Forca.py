from random import randint
palavras0=['hotel','shopping','aeroporto','escola','museu','faculdade']
palavras1=['macarrao','bacalhau','bruschetta','pizza','carne', 'esfiha']
palavras2=['A Chegada','Interestelar','Homem de Ferro','Star Wars','Blade Runner', 'O Poderoso Chefao']
palavras3=['Big Bang theory','Brooklyn nine nine','Game of thrones','House of cards','Familia sopranos', 'The office']
palavras4=['computador','celular','smartwatch','tablet','televisao','secretaria de IA']
L=[palavras0,palavras1,palavras2,palavras3,palavras4]
a=randint(0, 4)
if a==0:
    print('O tema da forca é lugar')
if a==1:
    print('O tema da forca é comida')
if a==2:
    print('O tema da forca é filme')
if a==3:
    print('O tema da forca é serie')
if a==4:
    print('O tema da forca é tecnologia')
b=randint(0, 5)
L0=L[a][b]
tamanho=['_']*(len(L0))
x=len(L0)+4
while x>0:
  print(' '.join(tamanho), f'({x})')
  letra=intput('Digite uma letra: ')
  for i in range(len(tamanho)):
    if L0[i]==letra:
      tamanho[i]=letra
  if ''.join(L0)==tamanho:
    print('Parabéns, você acertou a palavra: ', ''.join(L0))
    break
  else:
    print('Infelizmente, você errou a palavra: ', ''.join(L0))
    break
  x-=1
