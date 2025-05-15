import pygame, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from collections import deque

pygame.init()

LARGURA_TELA, ALTURA_TELA, tela, BRANCO, PRETO, fonte=500, 800, pygame.display.set_mode((500, 800)), (255, 255, 255), (0, 0, 0), pygame.font.SysFont("Arial", 24)
pygame.display.set_caption("Flappy Bird RL")

class Passaro:
    def __init__(self):
        self.rect=pygame.Rect(50, ALTURA_TELA // 2, 30, 30)
        self.vel=0

    def pular(self):
        self.vel=-9

    def atualizar(self):
        self.vel, self.rect.y+=1, self.vel

class Cano:
    def __init__(self):
        self.largura, self.espaco, self.x, self.topo=60, 200, LARGURA_TELA, random.randint(50, ALTURA_TELA - self.espaco - 50)

    def atualizar(self):
        self.x-=5

    def retangulos(self):
        topo_rect=pygame.Rect(self.x, 0, self.largura, self.topo)
        base_rect=pygame.Rect(self.x, self.topo + self.espaco, self.largura, ALTURA_TELA - self.topo - self.espaco)
        return topo_rect, base_rect

class FlappyBirdEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.passaro, self.canos, self.frame, self.pontos=Passaro(), [Cano()], 0, 0
        return self.obter_estado()

    def step(self, acao):
        recompensa=-0.1
        self.frame+=1

        if acao==1:
            self.passaro.pular()

        self.passaro.atualizar()

        remover_canos=[]
        adicionar_cano=False

        for cano in self.canos:
            cano.atualizar()
            if cano.x+cano.largura < 0:
                remover_canos.append(cano)

            if cano.x + cano.largura < self.passaro.rect.x and not hasattr(cano, "passou"):
                cano.passou=True
                self.pontos+=1
                recompensa+=10
                adicionar_cano=True

        if adicionar_cano:
            self.canos.append(Cano())

        for cano in remover_canos:
            self.canos.remove(cano)

        for cano in self.canos:
            topo, base = cano.retangulos()
            if self.passaro.rect.colliderect(topo) or self.passaro.rect.colliderect(base):
                return self.obter_estado(), -10, True, {}

        if self.passaro.rect.top <= 0 or self.passaro.rect.bottom >= ALTURA_TELA:
            return self.obter_estado(), -10, True, {}

        return self.obter_estado(), recompensa, False, {}

    def render(self):
        tela.fill(BRANCO)
        pygame.draw.rect(tela, PRETO, self.passaro.rect)
        for cano in self.canos:
            topo, base=cano.retangulos()
            pygame.draw.rect(tela, PRETO, topo)
            pygame.draw.rect(tela, PRETO, base)
        texto=fonte.render(f"Pontos: {self.pontos}", True, PRETO)
        tela.blit(texto, (10, 10))
        pygame.display.flip()

    def obter_estado(self):
        cano_proximo=None
        for cano in self.canos:
            if cano.x+cano.largura > self.passaro.rect.x:
                cano_proximo=cano
                break

        if not cano_proximo:
            cano_proximo=Cano()

        dist_x=cano_proximo.x-self.passaro.rect.x
        dist_y=(cano_proximo.topo+cano_proximo.espaco/2)-self.passaro.rect.y

        return np.array([self.passaro.rect.y / ALTURA_TELA,np.clip(self.passaro.vel, -20, 20) / 20,dist_x / LARGURA_TELA,dist_y / ALTURA_TELA], dtype=np.float32)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, 128),nn.ReLU(),nn.Linear(128, 128),nn.ReLU(),nn.Linear(128, output_dim))
    def forward(self, x):
        return self.fc(x)

class Agente:
    def __init__(self, state_size, action_size):
        self.state_size=state_size
        self.action_size=action_size
        self.memory=deque(maxlen=2000)
        self.gamma=0.99
        self.epsilon=1.0
        self.epsilon_min=0.05
        self.epsilon_decay=0.995
        self.model=DQN(state_size, action_size)
        self.criterion=nn.MSELoss()
        self.optimizer=optim.Adam(self.model.parameters(), lr=1e-4)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values=self.model(state)
        return torch.argmax(q_values).item()

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch=random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor=torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor=torch.FloatTensor(next_state).unsqueeze(0)
            target=reward
            if not done:
                target+=self.gamma * torch.max(self.model(next_state_tensor)).item()
            q_values = self.model(state_tensor)
            target_q_values = q_values.clone().detach()
            target_q_values[0][action] = target
            loss = self.criterion(q_values, target_q_values)
            loss.backward()
            self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon*=self.epsilon_decay

EPISODIOS, env, agente, batch_size=3000, FlappyBirdEnv(), Agente(state_size=4, action_size=2), 32

for e in range(EPISODIOS):
    estado=env.reset()
    total_recompensa=0
    for t in range(1000):
        pygame.event.pump()
        acao=agente.act(estado)
        proximo_estado, recompensa, feito, _=env.step(acao)
        agente.memorize(estado, acao, recompensa, proximo_estado, feito)
        estado=proximo_estado
        total_recompensa+=recompensa
        if feito:
            break
        if t % 4 == 0:
            env.render()
    agente.replay(batch_size)
    if e > 50:
        agente.update_epsilon()
    print(f"Episódio {e + 1}: Pontos = {env.pontos}, Epsilon = {agente.epsilon:.3f}, Recompensa = {total_recompensa:.2f}")
pygame.quit()
