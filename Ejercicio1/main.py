"""
Perceptrón con Agentes usando Mesa 0.8.9 + Interfaz con Matplotlib (en español)

Requisitos:
- mesa==0.8.9
- matplotlib
- numpy

Ejecución:
$ python -m pip install -r requirements.txt
$ python main.py

Interfaz (en español):
- Slider: Tasa de aprendizaje
- Slider: Iteraciones (épocas)
- Botones: Iniciar entrenamiento, Restablecer pesos, Regenerar datos
- Visualización en tiempo real: puntos y línea de decisión
- Después del entrenamiento se muestra la precisión en un conjunto de prueba
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Mesa 0.8.9 (sin fallback)
from mesa import Agent, Model
from mesa.time import BaseScheduler


# ----------------------
# Agente: Punto (dato)
# ----------------------
class PointAgent(Agent):
    """
    Agente que representa un punto en el plano 2D.
    Atributos:
    - pos: (x, y)
    - label: 1 o -1 (según la línea separadora verdadera)
    - predicted: predicción actual del perceptrón (1 o -1)
    - correct: bool indicando si la predicción coincide con label
    """
    def __init__(self, unique_id, model, x, y, label):
        super().__init__(unique_id, model)
        self.pos = (x, y)
        self.label = label
        self.predicted = None
        self.correct = False

    def step(self):
        # Actualizar la predicción usando el perceptrón del modelo
        out = self.model.predict(self.pos)
        self.predicted = out
        self.correct = (self.predicted == self.label)


# ----------------------
# Modelo: Perceptrón con agentes
# ----------------------
class PerceptronModel(Model):
    """
    Modelo que contiene:
    - agentes que representan puntos de datos
    - pesos y bias del perceptrón
    - método de entrenamiento (una pasada por todo el dataset = 1 época)
    """
    def __init__(self, n_points=100, lr=0.1, seed=None, train_ratio=0.7):
        super().__init__()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.schedule = BaseScheduler(self)
        self.n_points = n_points
        self.lr = lr

        # Generar una línea separadora verdadera aleatoria: y = m*x + c
        self._true_m = random.uniform(-1.5, 1.5)
        self._true_c = random.uniform(-0.3, 0.3)

        # Generar dataset (points) y etiquetas según la línea verdadera
        points = []
        for i in range(n_points):
            x = random.uniform(-1.0, 1.0)
            y = random.uniform(-1.0, 1.0)
            label = 1 if y > (self._true_m * x + self._true_c) else -1
            agent = PointAgent(i, self, x, y, label)
            # En BaseScheduler de Mesa 0.8.9 el método para añadir agente es add()
            self.schedule.add(agent)
            points.append(agent)

        # Separar en entrenamiento y prueba
        random.shuffle(points)
        split = int(train_ratio * n_points)
        self.train_agents = points[:split]
        self.test_agents = points[split:]

        # Inicializar pesos y bias aleatoriamente (2 entradas + bias)
        self.weights = np.random.uniform(-1, 1, size=2)
        self.bias = np.random.uniform(-1, 1)

        # Estado del entrenamiento
        self.epoch = 0

    def predict_raw(self, xy):
        """Retorna el valor real w·x + b (float)"""
        x_vec = np.array([xy[0], xy[1]])
        return np.dot(self.weights, x_vec) + self.bias

    def predict(self, xy):
        """Retorna la predicción en {1, -1}"""
        val = self.predict_raw(xy)
        return 1 if val >= 0 else -1

    def step(self):
        """Un paso del scheduler: pedir a cada agente que actualice su predicción"""
        # BaseScheduler proporciona .agents (lista) en esta versión
        for agent in self.schedule.agents:
            agent.step()

    def train_one_epoch(self):
        """
        Una pasada completa sobre los datos de entrenamiento (una época).
        Regla de actualización del perceptrón:
            Si y != y_pred:
                w <- w + lr * y * x
                b <- b + lr * y
        """
        updated = 0
        for agent in self.train_agents:
            x = np.array([agent.pos[0], agent.pos[1]])
            y = agent.label
            pred = self.predict(agent.pos)
            if pred != y:
                # actualización
                self.weights += self.lr * y * x
                self.bias += self.lr * y
                updated += 1
        self.epoch += 1
        # Actualizar predicciones de todos los agentes
        self.step()
        return updated

    def evaluate(self, on_train=False):
        """Evalúa el desempeño. Si on_train True, evalúa sobre train_agents; si no, sobre test_agents."""
        data = self.train_agents if on_train else self.test_agents
        total = len(data)
        correct = 0
        for agent in data:
            pred = self.predict(agent.pos)
            if pred == agent.label:
                correct += 1
        return correct, total, (correct / total * 100.0 if total > 0 else 0.0)


# ----------------------
# Interfaz con Matplotlib (sliders y botones) - TODO en español
# ----------------------
class PerceptronGUI:
    def __init__(self, initial_points=200):
        # Parámetros iniciales
        self.n_points = initial_points
        self.learning_rate = 0.1
        self.iterations = 20  # número de épocas
        self.is_training = False
        self.model = PerceptronModel(n_points=self.n_points, lr=self.learning_rate, seed=42)

        # Crear figura y ejes
        # Elegir un estilo disponible para evitar OSError por estilos no instalados
        if 'seaborn-darkgrid' in plt.style.available:
            plt.style.use('seaborn-darkgrid')
        elif 'seaborn' in plt.style.available:
            plt.style.use('seaborn')
        else:
            plt.style.use('ggplot')

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(left=0.1, bottom=0.25)
        self.scatter = None
        self.line_plot = None
        self.text_accuracy = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes, fontsize=12, verticalalignment='top')

        # Límites visuales
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Perceptrón con Agentes (Mesa 0.8.9) - Visualización en tiempo real")

        # Widgets: sliders y botones (en español)
        axcolor = 'lightgoldenrodyellow'
        ax_lr = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=axcolor)
        ax_iters = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
        ax_start = plt.axes([0.1, 0.025, 0.22, 0.05])
        ax_reset = plt.axes([0.345, 0.025, 0.22, 0.05])
        ax_regen = plt.axes([0.59, 0.025, 0.32, 0.05])

        self.slider_lr = Slider(ax_lr, 'Tasa de aprendizaje', 0.001, 1.0, valinit=self.learning_rate, valstep=0.001)
        self.slider_iters = Slider(ax_iters, 'Iteraciones (épocas)', 1, 200, valinit=self.iterations, valstep=1)

        self.button_start = Button(ax_start, 'Iniciar entrenamiento', color='lightgreen', hovercolor='green')
        self.button_reset = Button(ax_reset, 'Restablecer pesos', color='lightcoral', hovercolor='red')
        self.button_regen = Button(ax_regen, 'Regenerar datos', color='lightblue', hovercolor='deepskyblue')

        # Conectar eventos
        self.slider_lr.on_changed(self.on_lr_change)
        self.slider_iters.on_changed(self.on_iters_change)
        self.button_start.on_clicked(self.on_start)
        self.button_reset.on_clicked(self.on_reset)
        self.button_regen.on_clicked(self.on_regenerate)

        # Dibujar estado inicial
        self.update_plot(initial=True)

    def on_lr_change(self, val):
        self.learning_rate = float(val)
        self.model.lr = self.learning_rate

    def on_iters_change(self, val):
        self.iterations = int(val)

    def on_start(self, event):
        if self.is_training:
            # Si ya está entrenando, ignorar
            return
        self.is_training = True
        # Intentar deshabilitar sliders mientras entrena (visual)
        try:
            self.slider_lr.set_active(False)
            self.slider_iters.set_active(False)
        except Exception:
            pass

        try:
            self.run_training()
        finally:
            try:
                self.slider_lr.set_active(True)
                self.slider_iters.set_active(True)
            except Exception:
                pass
            self.is_training = False

    def on_reset(self, event):
        # Reinicia pesos y bias aleatoriamente, no regenera datos
        self.model.weights = np.random.uniform(-1, 1, size=2)
        self.model.bias = np.random.uniform(-1, 1)
        self.model.epoch = 0
        self.model.step()  # actualizar predicciones
        self.update_plot()

    def on_regenerate(self, event):
        # Regenerar todo el dataset y reiniciar pesos
        self.model = PerceptronModel(n_points=self.n_points, lr=self.learning_rate, seed=random.randint(0, 9999))
        self.update_plot()

    def update_plot(self, initial=False):
        # Obtener coordenadas y colores según si están correctamente clasificados
        xs = []
        ys = []
        colors = []
        sizes = []
        for agent in self.model.schedule.agents:
            xs.append(agent.pos[0])
            ys.append(agent.pos[1])
            # actualizamos el estado de predicción antes de colorear
            pred = self.model.predict(agent.pos)
            correct = (pred == agent.label)
            colors.append('green' if correct else 'red')
            sizes.append(40 if agent in self.model.train_agents else 20)  # entrenamiento más grande

        # Limpiar y volver a dibujar
        self.ax.cla()
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        # Título en español mostrando la época actual
        self.ax.set_title(f"Perceptrón - Época {self.model.epoch}")

        # Dibujar puntos
        self.scatter = self.ax.scatter(xs, ys, c=colors, s=sizes, edgecolor='k', alpha=0.9)

        # Dibujar la línea de separación verdadera (para referencia) en gris punteada
        xs_line = np.array([-1.2, 1.2])
        ys_true = self.model._true_m * xs_line + self.model._true_c
        self.ax.plot(xs_line, ys_true, linestyle='--', color='gray', label='Frontera verdadera')

        # Dibujar la línea de decisión actual del perceptrón
        w = self.model.weights
        b = self.model.bias
        # Manejar caso w[1] ~ 0 -> línea vertical
        if abs(w[1]) > 1e-6:
            ys_decision = -(w[0] / w[1]) * xs_line - (b / w[1])
            self.ax.plot(xs_line, ys_decision, color='blue', linewidth=2, label='Frontera del perceptrón')
        else:
            # Línea vertical x = -b/w0
            if abs(w[0]) > 1e-6:
                x_vert = -b / w[0]
                self.ax.axvline(x=x_vert, color='blue', linewidth=2, label='Frontera del perceptrón (vertical)')
        # Leyenda en español
        self.ax.legend(loc='lower right')

        # Mostrar precisión en conjunto de prueba y entrenamiento (en español)
        correct_test, total_test, pct_test = self.model.evaluate(on_train=False)
        correct_train, total_train, pct_train = self.model.evaluate(on_train=True)
        self.text_accuracy = self.ax.text(0.02, 0.95,
                                          f"Entrenamiento: {correct_train}/{total_train} = {pct_train:.1f}%\n"
                                          f"Prueba:       {correct_test}/{total_test} = {pct_test:.1f}%",
                                          transform=self.ax.transAxes, fontsize=12, verticalalignment='top',
                                          bbox=dict(facecolor='white', alpha=0.6))
        # Redibujar canvas
        self.fig.canvas.draw_idle()
        if initial:
            plt.show(block=False)

    def run_training(self):
        """
        Ejecuta el entrenamiento durante self.iterations épocas.
        Se actualiza la visualización tras cada época para mostrar la línea y colores.
        """
        iterations = self.iterations
        # Bucle de entrenamiento con pausas para actualizar la figura
        for it in range(iterations):
            updated = self.model.train_one_epoch()
            # Actualizar display
            self.update_plot()
            # Pequeña pausa para permitir la actualización visual
            plt.pause(0.15)
            # Si no hubo actualizaciones (converge), podemos romper antes
            if updated == 0:
                # convergió perfectamente en el conjunto de entrenamiento
                break

        # Al finalizar, calcular y mostrar evaluación final (mensaje en español)
        correct_test, total_test, pct_test = self.model.evaluate(on_train=False)
        correct_train, total_train, pct_train = self.model.evaluate(on_train=True)
        print(f"Entrenamiento finalizado en época {self.model.epoch}. Precisión (entrenamiento): {pct_train:.2f}% | Precisión (prueba): {pct_test:.2f}%")
        # Actualizar la figura final
        self.update_plot()


# ----------------------
# Ejecutar la GUI
# ----------------------
if __name__ == "__main__":
    gui = PerceptronGUI(initial_points=250)
    # Mantener la ventana abierta
    plt.show()