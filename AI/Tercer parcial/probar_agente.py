import gym
from QLearn import Agente_Q
import torch

env = gym.make('Pong-ram-v4',
               #frameskip=8,
               render_mode='human')

agente = Agente_Q(env)
agente.cargar("AI/Tercer parcial/agentes_q", "pong")

estado = env.reset()
for _ in range(300):
    estado = torch.tensor([estado], dtype=torch.float) / 255
    action = agente.seleccionar_accion(estado, modo_eval=True)
    estado, reward, done, info = env.step(action.item())
    #print(action, reward)
    if done:
        estado = env.reset()
env.close()