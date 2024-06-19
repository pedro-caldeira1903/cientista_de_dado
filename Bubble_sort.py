from random import randint
def gera(tamanho):
  L=[]
  for i in range(tamanho):
    L.append(randint(1, 99))
  return L
L0=gera(10)
print(L0)
def bubble_sort(L):
  for j in range(len(L), 1, -1):
    ordenacao=True
    for i in range(j-1):
      if L[i]>L[i+1]:
        ordenacao=False
        temp=L[i]
        L[i]=L[i+1]
        L[i+1]=temp
    while ordenacao:
      break
bubble_sort(L0)
print(L0)
