import gym
from QLearn import Agente_Q
import torch

env = gym.make('Breakout-ram-v0',
            frameskip=5,
            render_mode='human')

agente = Agente_Q(env, max_mem=1000)
agente.cargar("AI/Tercer parcial/agentes_q", "agente1")

estado = env.reset()
for _ in range(300):
    #env.render()
    estado = torch.tensor([estado], dtype=torch.float)
    action = agente.seleccionar_accion(estado)
    estado, reward, done, info = env.step(action)
    if done:
        estado = env.reset()
env.close()