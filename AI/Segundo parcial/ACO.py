import numpy as np
from grafos import Grafo


class Ant():

    def __init__(self):
        self.recorrido = []
        self.dist_recorrida = 0

    def iniciar(self, nodo):
        self.__init__()
        self.mover(nodo, 0)

    def mover(self, nodo, distancia):
        self.recorrido.append(nodo)
        self.dist_recorrida += distancia


class Colony():

    def __init__(self, grafo, n_ants):
        self.grafo = grafo
        # Grafo de feromonas con la topología del grafo de entrada
        self.feromonas = grafo.copia()
        # Instanciar hormigas de la colonia
        self.ants = [Ant() for _ in range(n_ants)]


    def ruta_min(self, origen, destino, n_epocas=100,
                 alfa=0.5, beta=0.5, p=0.1, metrica='estaciones'):
        """
        Devuelve la secuencia de nodos de la ruta que minimiza el
        número de estaciones recorridas o el número de transbordos
        necesarios, según la métrica utilizada.
        """
        mejor_distancia = np.inf
        mejor_recorrido = []
        # Iniciar feromonas con un valor ligeramente positivo
        self.feromonas.transformar_bordes(lambda _: 0.01)
        for _ in range(n_epocas):
            for ant in self.ants:
                ant.iniciar(origen)
                # Recorrido de la hormiga
                nodo = origen
                while nodo != destino:
                    vecinos = self.grafo.vecinos(nodo)
                    probs = self._prob_transicion(ant.recorrido, vecinos,
                                                  alfa, beta)
                    # Escoger siguiente nodo en función de probabilidades
                    nodo_nuevo = np.random.choice(vecinos, p=probs)
                    distancia = self.grafo.valor_borde(nodo, nodo_nuevo)
                    ant.mover(nodo_nuevo, distancia)
                    nodo = nodo_nuevo

                # Definir tipo de distancia
                if metrica == 'estaciones':
                    distancia = ant.dist_recorrida
                elif metrica == 'transbordos':
                    distancia = len(ant.recorrido)
                else:
                    raise ValueError('Métricas válidas: transbordos, estaciones')

                # Colocar feromonas en el recorrido
                feromona = 1/distancia
                self._propagar_feromonas(ant.recorrido, feromona)
                # Actualizar registro de mejores valores
                if distancia < mejor_distancia:
                    mejor_recorrido = ant.recorrido
                    mejor_distancia = distancia
            # Evaporar feromonas
            self.feromonas.transformar_bordes(lambda x: (1-p)*x)

        # Devolver secuencia
        return mejor_recorrido


    def _prob_transicion(self, recorrido, vecinos, alfa, beta):
        # Devolver vector con probabilidad de cada borde disponible
        probs = []
        nodo_actual = recorrido[-1]
        for vecino in vecinos:
            feromona = self.feromonas.valor_borde(nodo_actual, vecino)
            distancia = self.grafo.valor_borde(nodo_actual, vecino)
            p = feromona**alfa * distancia**beta
            # Eliminar probabilidad de regresar al nodo anterior
            if len(recorrido) > 1:
                nodo_previo = recorrido[-2]
                if vecino == nodo_previo:
                    p = 0
            probs.append(p)
        # Normalizar probabilidad
        probs = [p/sum(probs) for p in probs]
        return probs


    def _propagar_feromonas(self, recorrido, feromona):
        # Iterar recorrido por pares consecutivos de nodos
        for nodo1, nodo2 in zip(recorrido, recorrido[1:]):
            ferom_prev = self.feromonas.valor_borde(nodo1, nodo2)
            self.feromonas.asignar_borde(nodo1, nodo2, feromona + ferom_prev)



if __name__ == '__main__':

    from grafos import Metro

    metro = Metro()
    col = Colony(metro, 10)

    print(col.ruta_min('El Rosario', 'Pino Suárez', metrica='transbordos'))
    #print(col.ruta_min('Ermita', 'Deportivo 18 de Marzo', metrica='estaciones'))