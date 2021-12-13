import random
from collections import namedtuple, deque
from itertools import count
import os
import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, state_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, action_size)
        )

    def forward(self, x):
        return self.model(x)

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

class Agente_Q():
    """docstring for ClassName."""
    def __init__(self, env, max_mem):
        self.env = env
        self.memoria = Memoria(max_mem)
        self.n_acciones = self.env.action_space.n
        self.policy_net = DQN(128, self.n_acciones)
        self.target_net = DQN(128, self.n_acciones)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
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

    def seleccionar_accion(self, state):
        eps_inicial = 0.9
        eps_final = 0.05
        eps_decay = 200

        eps_threshold = eps_final + (eps_inicial - eps_final) * \
            np.exp(-1. * self.steps_done / eps_decay)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_acciones)]], dtype=torch.long)
            #return self.env.action_space.sample()


    def entrenar(self, n_episodios, batch_size, gamma, target_update=5):
        for i_episode in range(n_episodios):
            # Initialize the environment and state
            state = self.env.reset()
            state = torch.tensor([state], dtype=torch.float)
            #state = self.env.step(0)[0]
            for t in count(): #while True: ?
                # Select and perform an action
                action = self.seleccionar_accion(state)
                next_state, reward, done, _ = self.env.step(action.item())
                next_state = torch.tensor([next_state], dtype=torch.float)
                reward = torch.tensor([reward])
                # Hacer iguales los estados terminales
                if done:
                    next_state = None

                # Store the transition in memory
                self.memoria.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state
                # Paso de optimización de policy_net
                self._optimize_model(batch_size, gamma)
                
                if done:
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())


    def _optimize_model(self, batch_size, gamma):
        """
        Paso de optimización de policy_net
        """
        if len(self.memoria) < batch_size:
            return
        transitions = self.memoria.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)),
                                            dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        #criterio = nn.HuberLoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Recorte de gradientes para restringirlos a [-1, 1]
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

if __name__ == "__main__":
    
    import gym

    env = gym.make('Breakout-ram-v0',
                frameskip=5)
                #render_mode='human')

    # observation = env.reset()
    # for _ in range(150):
    #     #env.render()
    #     action = env.action_space.sample() # your agent here (this takes random actions)
    #     observation, reward, done, info = env.step(action)
    #     print(reward)
    #     if done:
    #         observation = env.reset()
    # env.close()


    agente = Agente_Q(env, max_mem=1000)
    agente.entrenar(n_episodios=10, batch_size=16, gamma=0.5)
    agente.guardar("AI/Tercer parcial/agentes_q", "agente1")