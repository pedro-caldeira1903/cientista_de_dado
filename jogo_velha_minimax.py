import numpy as np
def minimax(tabuleiro, profundidade, maximizar_jogador):
  if verifica_vencedor(tabuleiro, 'x'):
    return -1
  elif verifica_vencedor(tabuleiro, 'o'):
    return 1
  else:
    return 0

  if maximizar_jogador:
    max_eval = float('-inf')
    for i in range(9):
      if tabuleiro[i]== '_':
        tabuleiro[i]= 'o'
        eval = minimax(tabuleiro, profundidade + 1, False)
        tabuleiro[i]='_'
        max_eval=max(max_eval, eval)
    return max_eval
  else:
    min_eval = float('inf')
    for i in range(9):
      if tabuleiro[i]=='_':
        tabuleiro[i]= 'x'
        eval = minimax(tabuleiro, profundidade + 1, True)
        tabuleiro[i]='_'
        min_eval=min(min_eval, eval)
    return min_eval
def melhor_jogada(tabuleiro):
  melhor_eval = float('-inf')
  jogada=None
  for i in range(9):
    if tabuleiro[i]=='_':
      tabuleiro[i]='o'
      eval = minimax(tabuleiro, 0, False)
      tabuleiro[i]='_'
      if eval > melhor_eval:
        melhor_eval = eval
        jogada=i
  return jogada
def verifica_vencedor(tabuleiro, jogador):
  if (tabuleiro[0]==jogador and tabuleiro[1]==jogador and tabuleiro[2]==jogador) or (tabuleiro[3]==jogador and tabuleiro[4]==jogador and tabuleiro[5]==jogador) or (tabuleiro[6]==jogador and tabuleiro[7]==jogador and tabuleiro[8]==jogador) or (tabuleiro[0]==jogador and tabuleiro[3]==jogador and tabuleiro[6]==jogador) or (tabuleiro[1]==jogador and tabuleiro[4]==jogador and tabuleiro[7]==jogador) or (tabuleiro[2]==jogador and tabuleiro[5]==jogador and tabuleiro[8]==jogador) or (tabuleiro[0]==jogador and tabuleiro[4]==jogador and tabuleiro[8]==jogador) or (tabuleiro[2]==jogador and tabuleiro[4]==jogador and tabuleiro[6]==jogador):
    return True
def tabuleiro_cheio(tabuleiro):
  return not any('_' in linha for linha in tabuleiro)
def exibir(tabuleiro):
  print(' '.join(L[0:3]))
  print(' '.join(L[3:6]))
  print(' '.join(L[6:9]))
def jogar():
  L=['_']*9
  exibir(L)
  while True:
    a=int(input('Digite o sua jogada:'))
    if L[a-1]=='_':
      L[a-1]='x'
    if verifica_vencedor(L, 'x'):
      exibir(L)
      print('Parabéns o humano ganhou!')
      break
    if tabuleiro_cheio(L):
      exibir(L)
      print('Empatou!')
      break
    b=melhor_jogada(L)
    L[b]='o'
    exibir(L)
    if verifica_vencedor(L, 'o'):
      exibir(L)
      print('Parabéns o computador ganhou!')
      break
