import random
from collections import namedtuple, deque
import os
import numpy as np
import torch
import torch.nn as nn



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class Memoria(object):

    def __init__(self, max_mem):
        self.memory = deque([],maxlen=max_mem)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """
    Red neuronal para aproximar una función Q
    """
    def __init__(self, state_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            # nn.Linear(16, 16),
            # nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, action_size)
        )

    def forward(self, x):
        return self.model(x)


class Agente_Q():

    def __init__(self, env, max_mem=1000):
        self.env = env
        self.memoria = Memoria(max_mem)
        self.n_acciones = self.env.action_space.n
        self.policy_net = DQN(128, self.n_acciones)
        self.target_net = DQN(128, self.n_acciones)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.steps_done = 0

    def guardar(self, path, name):
        # Almacenar los parámetros de policy_net
        name += ".pt"
        if not os.path.exists(path):
            print("Folder no existe, se creará uno nuevo")
            os.makedirs(path)
        save_path = os.path.join(path, name)
        torch.save(self.policy_net.state_dict(), save_path)

    def cargar(self, path, name):
        # Recuperar parámetros de un entrenamiento anterior
        name += ".pt"
        load_path = os.path.join(path, name)
        self.policy_net.load_state_dict(torch.load(load_path))

    def seleccionar_accion(self, state, eps_inicial=1.,
                           eps_final=0.1, eps_decay=200, modo_eval=False):
        # Estocástico al inicio, determinístico al final
        eps_threshold = eps_final + (eps_inicial - eps_final) * \
            np.exp(-1. * self.steps_done / eps_decay)
        self.steps_done += 1
        if random.random() > eps_threshold or modo_eval:
            # Selección determinística
            with torch.no_grad():
                # Devolver el índice(acción) con mayor valor
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Selección aleatoria
            return torch.tensor([[random.randrange(self.n_acciones)]], dtype=torch.long)


    def entrenar(self, n_episodios, batch_size, gamma, lr, target_update=10, 
                 eps_inicial=1., eps_final=0.1, eps_decay=200):

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        for i_episode in range(n_episodios):
            # Inicializar ambiente y estados
            state = self.env.reset()
            state = torch.tensor([state], dtype=torch.float) / 255
            while True:
                # Seleccionar acción y ejecutarla en el ambiente
                action = self.seleccionar_accion(state, eps_inicial, eps_final, eps_decay)
                next_state, reward, done, _ = self.env.step(action.item())
                # Acondicionar los datos como tensores
                next_state = torch.tensor([next_state], dtype=torch.float) / 255
                reward = torch.tensor([reward])
                # Hacer iguales los estados terminales
                if done:
                    next_state = None

                # Almacenar la transición de estados en memoria
                self.memoria.push(state, action, next_state, reward)

                # Transicionar de estado
                state = next_state
                # Paso de optimización de policy_net
                self._optimize_model(batch_size, gamma)
                
                if done:
                    break
            # Actualización periódica de target_net con los parámetros de policy_net
            if i_episode % target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())


    def _optimize_model(self, batch_size, gamma):
        """
        Paso de optimización de policy_net
        """
        if len(self.memoria) < batch_size:
            return
        transitions = self.memoria.sample(batch_size)
        # Acomodar los datos de memoria por batches
        batch = Transition(*zip(*transitions))

        # Máscara para filtrar los estados terminales
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)),
                                            dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # La predicción de policy_net de las mejores acciones para cada estado en los ejemplos
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Valores esperados de cada acción según target_net
        next_state_values = torch.zeros(batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Actualización de Bellman de los valores esperados
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Calcular pérdida y retropropagarla a los parámetros de policy_net
        criterion = nn.HuberLoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        # Recorte de gradientes para restringirlos a [-1, 1]
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        # Actualizar parámetros
        self.optimizer.step()


if __name__ == "__main__":
    
    import gym

    env = gym.make('Pong-ram-v4',
                   frameskip=4)

    agente = Agente_Q(env, max_mem=500)
    agente.entrenar(n_episodios=100, batch_size=128,
                    gamma=0.999, lr=1e-3, target_update=10, eps_decay=100)
    agente.guardar("AI/Tercer parcial/agentes_q", "pong")