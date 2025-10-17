# Informe: Perceptrón implementado con Agentes



Diseño — entrenamiento del perceptrón
---------------------
<img width="636" height="808" alt="image" src="https://github.com/user-attachments/assets/7631918c-9479-492f-a5a4-3feffac94944" />


Fundamento teórico
---------------------

1. Modelo del perceptrón
- Entrada: vector x ∈ R^2 (aquí: coordenadas (x, y)).
- Parámetros: pesos w = [w1, w2] y sesgo b.
- Función lineal: z = w · x + b.
- Activación (clasificación): ŷ = sign(z) -> devuelve 1 si z ≥ 0, -1 si z < 0.
- Es un clasificador lineal; sólo puede resolver problemas linealmente separables.

2. Regla de aprendizaje (Perceptron Learning Rule)
- Para cada muestra (x, y) con etiqueta y ∈ {1, −1}:
  - Calcular predicción ŷ = sign(w·x + b).
  - Si ŷ ≠ y, actualizar:
    - w ← w + η * y * x
    - b ← b + η * y
  - η (eta) es la tasa de aprendizaje (learning rate).
- El algoritmo iterativamente recorre el conjunto de entrenamiento (epocas) hasta convergencia (ninguna actualización en una época) o hasta alcanzar el número máximo de épocas.

---

Implementación
-----------------

 Estructura del proyecto (archivos principales)
- main.py — script principal que contiene:
  - Definición del agente (PointAgent).
  - Definición del modelo (PerceptronModel).
  - Interfaz gráfica y lógica de entrenamiento (PerceptronGUI).

 Generación de datos
- Se genera una línea separadora verdadera aleatoria: y = m*x + c, con m y c muestreados aleatoriamente en un rango.
- Se generan N puntos uniformes en el cuadrado [-1,1] × [-1,1].
- Etiqueta: label = 1 si y > m*x + c, else -1.
- Se divide el conjunto en entrenamiento y prueba (p. ej. 70/30 por defecto).

Clases principales 
- PointAgent(Agent)
  - Atributos: pos (x,y), label, predicted, correct.
  - Método step(): actualiza predicted y correct consultando model.predict(pos).
- PerceptronModel(Model)
  - Atributos: schedule (BaseScheduler), weights (w), bias (b), train_agents, test_agents, epoch.
  - Métodos:
    - predict_raw(xy): retorna w·x + b.
    - predict(xy): retorna 1 o -1 según signo.
    - train_one_epoch(): recorre train_agents y aplica regla de actualización; retorna número de actualizaciones.
    - evaluate(on_train): calcula aciertos y % de precisión sobre train/test.
- PerceptronGUI
  - Controles: slider tasa de aprendizaje, slider iteraciones, botones (Iniciar entrenamiento, Restablecer pesos, Regenerar datos).
  - update_plot(): dibuja puntos, frontera verdadera (gris punteada), frontera del perceptrón (azul), y cuadro con precisión.
  - run_training(): bucle que ejecuta train_one_epoch() por epoch, actualiza la visualización y opcionalmente guarda capturas.


--------------------------

Experimento realizado

 Parámetros usados (ejemplo práctico)
- Preset “Rápido / demostración” (ejemplo usado para capturas iniciales):
  - learning_rate = 0.001
  - iterations = 100
  - n_points = 250
  - train/test split = 70/30
  - seed = 42

<img width="1067" height="574" alt="image" src="https://github.com/user-attachments/assets/2b1b1983-ad19-4a61-9f9d-1f7802bc3bc0" />
<img width="1049" height="563" alt="image" src="https://github.com/user-attachments/assets/2ebef328-81db-4195-828f-5437328aaeb1" />


