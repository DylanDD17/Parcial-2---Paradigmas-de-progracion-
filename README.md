# Parcial-2---Paradigmas-de-progracion-

Descripción
------------
Este repositorio contiene las soluciones de los 3 ejercicios del Parcial 2 de la asignatura "Paradigmas de Programación". Incluye implementaciones en Python y Kotlin por ejercicio.

Requisitos generales
--------------------
- Python 3.8+ y pip
- Para los ejercicios en Python que usan interfaces gráficas: un entorno con soporte de Tk (suele venir en instalaciones estándar)
- Kotlin 1.9.10+ (para el ejercicio 3)
- JVM 17+ (para ejecutar Kotlin/JVM)
- Gradle 7.0+ (opcional, si el proyecto Kotlin está configurado con Gradle)

Instrucciones por ejercicio
---------------------------

Ejercicio 1
- Objetivo: Ejecutar el script principal (visualización con Mesa).
- Requisitos: mesa==0.8.9
- Instalación y ejecución:
```bash
pip install mesa==0.8.9
python main.py
```
- Nota: Ejecuta estos comandos en el directorio donde se encuentre `main.py`. Si aparece algún error relacionado con dependencias, crea un entorno virtual antes de instalar.

Ejercicio 2
- Objetivo: Ejecutar la calculadora (usa Mesa, numpy y matplotlib; puede requerir Tk para interfaces).
- Requisitos: mesa==0.8.9, numpy, matplotlib, tk
- Instalación y ejecución:
```bash
pip install mesa==0.8.9 numpy matplotlib tk
python calculadora.py
```
- Nota: En sistemas Linux, el paquete de Tk puede ser parte del sistema (ej. `sudo apt install python3-tk` en Debian/Ubuntu).

Ejercicio 3
- Objetivo: Código en Kotlin (proyecto JVM).
- Requisitos (mínimos):
  - Kotlin 1.9.10+
  - JVM 17+
  - Gradle 7.0+ (opcional si usas Gradle)
- Ejecución (ejemplo con Gradle, si el proyecto está configurado):
```bash
cd kotlin
./gradlew run        
./gradlew test
```
