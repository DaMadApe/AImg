

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
        existe_n1 = nodo1 in self._grafo
        existe_n2 = nodo2 in self._grafo
        assert existe_n1 and existe_n2, 'Nodos no han sido agregados'
        self._grafo[nodo1][nodo2] = d
        self._grafo[nodo2][nodo1] = d

    def nodos(self):
        # Devolver la lista de los nodos del grafo
        return list(self._grafo.keys())
    
    def vecinos(self, nodo):
        # Devolver la lista de nodos contiguos a un nodo
        assert nodo in self._grafo, 'Nodo no registrado'
        return list(self._grafo[nodo].keys())


class Metro(Grafo):

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

        # La distancia es la cantidad de estaciones intermedias
        self.asignar_borde('El Rosario', 'Instituto del Petróleo', 5)
        self.asignar_borde('El Rosario', 'Tacuba', 3)
        self.asignar_borde('Instituto del Petróleo', 'Deportivo 18 de Marzo', 1)
        self.asignar_borde('Instituto del Petróleo', 'La Raza', 1)
        self.asignar_borde('Deportivo 18 de Marzo', 'Martín Carrera', 1)
        self.asignar_borde('Martín Carrera', 'Consulado', 2)
        self.asignar_borde('Tacuba', 'Hidalgo', 6)
        self.asignar_borde('Tacuba', 'Tacubaya', 4)
        self.asignar_borde('La Raza', 'Consulado', 2)
        self.asignar_borde('La Raza', 'Guerrero', 1)
        self.asignar_borde('Consulado', 'Oceanía', 2)
        self.asignar_borde('Consulado', 'Morelos', 1)
        self.asignar_borde('Oceanía', 'Pantitlán', 2)
        self.asignar_borde('Guerrero', 'Garibaldi/Lagunilla', 0)
        self.asignar_borde('Guerrero', 'Hidalgo', 0)
        self.asignar_borde('Garibaldi/Lagunilla', 'Morelos', 2)
        self.asignar_borde('Garibaldi/Lagunilla', 'Bellas Artes', 0)
        self.asignar_borde('Morelos', 'San Lázaro', 0)
        self.asignar_borde('Morelos', 'Candelaria', 0)
        self.asignar_borde('Hidalgo', 'Bellas Artes', 0)
        self.asignar_borde('Hidalgo', 'Balderas', 1)
        self.asignar_borde('Bellas Artes', 'Pino Suárez', 2)
        self.asignar_borde('Bellas Artes', 'Salto del Agua', 1)
        self.asignar_borde('Balderas', 'Salto del Agua', 0)
        self.asignar_borde('Balderas', 'Centro Médico', 2)
        self.asignar_borde('Salto del Agua', 'Pino Suárez', 1)
        self.asignar_borde('Salto del Agua', 'Chabacano', 2)
        self.asignar_borde('Pino Suárez', 'Candelaria', 1)
        self.asignar_borde('Pino Suárez', 'Chabacano', 1)
        self.asignar_borde('Candelaria', 'San Lázaro', 0)
        self.asignar_borde('Candelaria', 'Jamaica', 1)
        self.asignar_borde('San Lázaro', 'Oceanía', 2)
        self.asignar_borde('San Lázaro', 'Pantitlán', 5)
        self.asignar_borde('Tacubaya', 'Centro Médico', 2)
        self.asignar_borde('Tacubaya', 'Mixcoac', 2)
        self.asignar_borde('Centro Médico', 'Chabacano', 1)
        self.asignar_borde('Centro Médico', 'Zapata', 3)
        self.asignar_borde('Chabacano', 'Jamaica', 0)
        self.asignar_borde('Chabacano', 'Santa Anita', 1)
        self.asignar_borde('Chabacano', 'Ermita', 5)
        self.asignar_borde('Jamaica', 'Pantitlán', 4)
        self.asignar_borde('Jamaica', 'Santa Anita', 0)
        self.asignar_borde('Santa Anita', 'Atlalilco', 5)
        self.asignar_borde('Mixcoac', 'Zapata', 2)
        self.asignar_borde('Zapata', 'Ermita', 2)
        self.asignar_borde('Ermita', 'Atlalilco', 1)


if __name__ == '__main__':

    metro = Metro()
    print(metro.vecinos('Jamaica'))