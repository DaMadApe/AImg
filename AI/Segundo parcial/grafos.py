"""
Clases auxiliares para representar grafos,
para usar con Ant Colony Optimization (ACO.py)
"""

class Grafo():

    def __init__(self):
        self._grafo = {}

    def copia(self):
        copia = Grafo()
        copia._grafo = self._grafo.copy()
        return copia

    def agregar_nodos(self, *nodos):
        for nodo in nodos:
            assert isinstance(nodo, str), 'Nombre de nodo debe ser string'
            self._grafo[nodo] = {}

    def asignar_borde(self, nodo1, nodo2, d):
        """
        Agregar una conexión con valor d entre nodos previamente agregados
        """
        existe_n1 = nodo1 in self._grafo
        existe_n2 = nodo2 in self._grafo
        assert existe_n1 and existe_n2, 'Nodos no han sido agregados'
        self._grafo[nodo1][nodo2] = d
        self._grafo[nodo2][nodo1] = d

    def valor_borde(self, nodo1, nodo2):
        return self._grafo[nodo1][nodo2]

    def nodos(self):
        # Devolver la lista de los nodos del grafo
        return list(self._grafo.keys())
    
    def vecinos(self, nodo):
        # Devolver la lista de nodos contiguos a un nodo
        assert nodo in self._grafo, 'Nodo no registrado'
        return list(self._grafo[nodo].keys())

    def transformar_bordes(self, f):
        """
        Aplica la función f a todos los valores de borde de la gráfica
        """
        for nodo in self._grafo.keys():
            for vecino in self._grafo[nodo].keys():
                self._grafo[nodo][vecino] = f(self._grafo[nodo][vecino])


class Metro(Grafo):
    """
    Instancia de grafo que representa el metro de la CDMX
    """
    def __init__(self):
        super().__init__()
        # Inicializar nodos: Estaciones de transbordo
        cruces = ['El Rosario', 'Instituto del Petróleo',
                  'Deportivo 18 de Marzo', 'Martín Carrera', 'Tacuba',
                  'La Raza', 'Consulado', 'Oceanía', 'Guerrero',
                  'Garibaldi/Lagunilla', 'Morelos', 'Oceanía', 'Hidalgo',
                  'Bellas Artes', 'Balderas', 'Salto del Agua', 'Pino Suárez',
                  'Candelaria', 'San Lázaro', 'Pantitlán', 'Tacubaya',
                  'Centro Médico', 'Chabacano', 'Jamaica', 'Santa Anita',
                  'Mixcoac', 'Zapata', 'Ermita', 'Atlalilco']
        self.agregar_nodos(*cruces)

        # La distancia es la cantidad de estaciones intermedias + 1
        self.asignar_borde('El Rosario', 'Instituto del Petróleo', 6)
        self.asignar_borde('El Rosario', 'Tacuba', 4)
        self.asignar_borde('Instituto del Petróleo', 'Deportivo 18 de Marzo', 2)
        self.asignar_borde('Instituto del Petróleo', 'La Raza', 2)
        self.asignar_borde('Deportivo 18 de Marzo', 'Martín Carrera', 2)
        self.asignar_borde('Martín Carrera', 'Consulado', 3)
        self.asignar_borde('Tacuba', 'Hidalgo', 7)
        self.asignar_borde('Tacuba', 'Tacubaya', 5)
        self.asignar_borde('La Raza', 'Consulado', 3)
        self.asignar_borde('La Raza', 'Guerrero', 2)
        self.asignar_borde('Consulado', 'Oceanía', 3)
        self.asignar_borde('Consulado', 'Morelos', 2)
        self.asignar_borde('Oceanía', 'Pantitlán', 3)
        self.asignar_borde('Guerrero', 'Garibaldi/Lagunilla', 1)
        self.asignar_borde('Guerrero', 'Hidalgo', 1)
        self.asignar_borde('Garibaldi/Lagunilla', 'Morelos', 3)
        self.asignar_borde('Garibaldi/Lagunilla', 'Bellas Artes', 1)
        self.asignar_borde('Morelos', 'San Lázaro', 1)
        self.asignar_borde('Morelos', 'Candelaria', 1)
        self.asignar_borde('Hidalgo', 'Bellas Artes', 1)
        self.asignar_borde('Hidalgo', 'Balderas', 2)
        self.asignar_borde('Bellas Artes', 'Pino Suárez', 3)
        self.asignar_borde('Bellas Artes', 'Salto del Agua', 2)
        self.asignar_borde('Balderas', 'Salto del Agua', 1)
        self.asignar_borde('Balderas', 'Centro Médico', 3)
        self.asignar_borde('Salto del Agua', 'Pino Suárez', 2)
        self.asignar_borde('Salto del Agua', 'Chabacano', 3)
        self.asignar_borde('Pino Suárez', 'Candelaria', 2)
        self.asignar_borde('Pino Suárez', 'Chabacano', 2)
        self.asignar_borde('Candelaria', 'San Lázaro', 1)
        self.asignar_borde('Candelaria', 'Jamaica', 2)
        self.asignar_borde('San Lázaro', 'Oceanía', 3)
        self.asignar_borde('San Lázaro', 'Pantitlán', 6)
        self.asignar_borde('Tacubaya', 'Centro Médico', 3)
        self.asignar_borde('Tacubaya', 'Mixcoac', 3)
        self.asignar_borde('Centro Médico', 'Chabacano', 2)
        self.asignar_borde('Centro Médico', 'Zapata', 4)
        self.asignar_borde('Chabacano', 'Jamaica', 1)
        self.asignar_borde('Chabacano', 'Santa Anita', 2)
        self.asignar_borde('Chabacano', 'Ermita', 4)
        self.asignar_borde('Jamaica', 'Pantitlán', 5)
        self.asignar_borde('Jamaica', 'Santa Anita', 1)
        self.asignar_borde('Santa Anita', 'Atlalilco', 6)
        self.asignar_borde('Mixcoac', 'Zapata', 3)
        self.asignar_borde('Zapata', 'Ermita', 3)
        self.asignar_borde('Ermita', 'Atlalilco', 2)


if __name__ == '__main__':

    grafo = Grafo()
    grafo.agregar_nodos('a', 'b', 'c', 'd')
    grafo.asignar_borde('a', 'b', 2)
    grafo.asignar_borde('b', 'c', 1)
    grafo.asignar_borde('b', 'd', 4)

    assert grafo.valor_borde('b', 'c') == 1