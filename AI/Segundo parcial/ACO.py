import numpy as np
from grafos import Grafo


class Ant():
    """
    
    """
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
    """
    
    """
    def __init__(self, grafo, n_ants):
        self.grafo = grafo
        self.feromonas = grafo.copia()
        self.ants = [Ant() for _ in range(n_ants)]

    def ruta_min(self, origen, destino, n_epocas=100,
                 alfa=0, beta=1, p=0.1):
        """
        Devuelve la secuencia de nodos de la ruta más corta
        """
        for epoca in range(n_epocas):
            for ant in self.ants:
                ant.iniciar(origen)
                nodo = origen
                while nodo != destino:
                    vecinos = self.grafo.vecinos(nodo)
                    probs = self._prob_transicion(nodo, vecinos, alfa, beta)
                    # Escoger siguiente nodo en función de probabilidades
                    nuevo_nodo = np.random.choice(vecinos, p=probs)
                    distancia = self.grafo.valor_borde(nodo, nuevo_nodo)
                    ant.mover(nuevo_nodo, distancia)
                    nodo = nuevo_nodo

                feromona = 1/ant.dist_recorrida
                self._propagar_feromonas(ant.recorrido, feromona)
            # Evaporar feromonas
            self.feromonas.transformar_bordes(lambda x: (1-p)*x)

        # Devolver secuencia con más feromonas
        ruta = origen
        return self.feromonas._grafo

    def _prob_transicion(self, nodo, vecinos, alfa, beta):
        # Devolver vector con probabilidad de cada borde disponible
        probs = []
        for vecino in vecinos:
            feromona = self.feromonas.valor_borde(nodo, vecino)
            distancia = self.grafo.valor_borde(nodo, vecino)
            p = feromona**alfa * distancia**beta
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

    grafo = Grafo()
    grafo.agregar_nodos('a', 'b', 'c', 'd', 'e')
    grafo.asignar_borde('a', 'b', 1)
    grafo.asignar_borde('b', 'c', 1)
    grafo.asignar_borde('b', 'd', 1)
    grafo.asignar_borde('a', 'c', 1)
    grafo.asignar_borde('c', 'e', 1)
    grafo.asignar_borde('d', 'e', 1)

    col = Colony(grafo, 10)

    print(grafo._grafo)
    print(col.ruta_min('a', 'e'))