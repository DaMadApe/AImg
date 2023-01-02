import numpy as np

class ABC():
    def __init__(self, n_fuentes, x_min, x_max, dim):
        self.n_fuentes = n_fuentes
        self.x_min = x_min
        self.x_max = x_max
        self.dim = dim

        self.fuentes = np.random.uniform(x_min, x_max, (n_fuentes,dim))
        self.valores = np.zeros(n_fuentes)
        self.explotados = np.zeros(n_fuentes, dtype=int)

    def buscar(self, funcion, max_iter=100, limit_exploit=10):
        self.f = funcion
        self.valores = [funcion(fuente) for fuente in self.fuentes]
        self.valores = np.array(self.valores)
        self.mejor_valor = np.inf
        self._registrar_mejor()
        for _ in range(max_iter):
            self._fase_empleadas()
            self._fase_observadoras()
            self._registrar_mejor()
            self._fase_exploradoras(limit_exploit)
        return self.mejor_sol, self.mejor_valor

    def _registrar_mejor(self):
        i = np.argmin(self.valores)
        if self.valores[i] < self.mejor_valor:
            self.mejor_valor = self.valores[i]
            self.mejor_sol = self.fuentes[i].copy()

    def _probar_nueva_solucion(self, indice, nueva_fuente):
        nueva_fuente = np.clip(nueva_fuente, self.x_min, self.x_max)
        valor_busqueda = self.f(nueva_fuente)
        if valor_busqueda < self.valores[indice]:
            self.fuentes[indice] = nueva_fuente
            self.valores[indice] = valor_busqueda
            self.explotados[indice] = 0
        else:
            self.explotados[indice] += 1

    def _fase_empleadas(self):
        # Para buscar, cada fuente se acerca a otra fuente aleatoria
        phi = np.random.uniform(-1,1, self.fuentes.shape)
        paso = phi*(self.fuentes - np.random.permutation(self.fuentes))
        busqueda = self.fuentes + paso
        for i in range(self.n_fuentes):
            self._probar_nueva_solucion(i, busqueda[i])

    def _fase_observadoras(self):
        fitness = -self.valores + self.valores.max() + 1
        probs = fitness/fitness.sum()
        for _ in range(self.n_fuentes):
            # Índice aleatorio en función de vector de probabilidades
            i = np.random.choice(np.arange(self.n_fuentes), p=probs)
            # Índice de solución aleatoria
            k = np.random.choice(np.arange(self.n_fuentes))
            # Nueva solución
            phi = np.random.uniform(-1, 1, self.dim)
            paso = phi*(self.fuentes[i] - self.fuentes[k])
            busqueda = self.fuentes[i] + paso
            self._probar_nueva_solucion(i, busqueda)

    def _fase_exploradoras(self, limit_exploit):
        #sobre_explotadas = self.explotados > limit_exploit
        i = np.argmax(self.explotados)
        if self.explotados[i] > limit_exploit:
            self.fuentes[i] = np.random.uniform(self.x_min,
                                                self.x_max,
                                                self.dim)
            self.valores[i] = self.f(self.fuentes[i])
            self.explotados[i] = 0

"""
Pruebas
"""
if __name__ == '__main__':

    def sphere(x):
        #min @ (0, 0) = 0
        return np.sum(x**2)

    def eggholder(x):
        #min @ (512, 404.2319) = -959.6407
        term1 = -(x[1]+47) * np.sin(np.sqrt(abs(x[0]/2 + x[1]+47)))
        term2 = -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]+47))))
        return term1 + term2

    colonia = ABC(n_fuentes=12, x_min=0, x_max=512, dim=2)
    print(colonia.buscar(eggholder))