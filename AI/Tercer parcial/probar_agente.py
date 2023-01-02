import gym
from QLearn import Agente_Q
import torch

env = gym.make('Pong-ram-v4',
               #frameskip=4,
               render_mode='human')

agente = Agente_Q(env)
agente.cargar("AI/Tercer parcial/agentes_q", "pong")

estado = env.reset()
for _ in range(1000):
    estado = torch.tensor([estado], dtype=torch.float) / 255
    # Modo_eval elimina decisiones estocásticas del agente
    action = agente.seleccionar_accion(estado, modo_eval=True)
    estado, reward, done, _ = env.step(action.item())
    print(reward)
    if done:
        estado = env.reset()
env.close()