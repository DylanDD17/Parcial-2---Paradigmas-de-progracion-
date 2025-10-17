# Informe

## Estructura principal

- **Calculadora (clase base):** operaciones básicas y manejo de memoria (implementa interface `Memorizable`).
- **CalculadoraCientifica (subclase):** añade funciones científicas, constantes y control de modo grados/radianes.
- **ExpressionEvaluator (shunting-yard → RPN):** convierte y evalúa expresiones; trabaja con una jerarquía interna de `Token` (sealed class).
- **Main (UI de consola):** interacción con el usuario, utiliza las clases anteriores.


---

## Encapsulamiento

### Campos privados
En `Calculadora` hay un campo `private var memory: Double` que no es accesible directamente desde fuera.  
El acceso/modificación se realiza exclusivamente mediante los métodos de la interfaz `Memorizable`:  
`memoryAdd`, `memorySubtract`, `memoryRecall`, `memoryClear`.  
Esto evita lectura/escritura directa y permite validar o cambiar la implementación interna sin afectar a los usuarios de la clase.

### Interfaz pública controlada
`Calculadora` expone métodos públicos para operaciones básicas: `sumar`, `restar`, `multiplicar`, `dividir`.  
La implementación interna usa `Double` y lanza `CalculatorException` en condiciones inválidas (por ejemplo, división por cero).  
De esta manera el comportamiento y las validaciones quedan centralizadas.

### Validaciones y manejo de errores
Métodos como `dividir()`, `raiz()`, `logaritmoBase10()`, `factorialInt()` validan sus entradas y lanzan `CalculatorException` con mensajes claros.  
Esto protege el estado interno y obliga al llamador a manejar errores en un punto controlado.

### Abstracción de constantes y comportamiento
`CalculadoraCientifica` expone `PI` y `E` mediante properties (`val PI: Double get() = Math.PI`).  
Internamente se puede cambiar la representación sin afectar a quien use la propiedad.


---

## Herencia

### Clase base: `Calculadora`
- Proporciona operaciones aritméticas básicas y la implementación de `Memorizable`.
- Declara métodos como `open fun sumar(...)`, `restar(...)`, `multiplicar(...)`, `dividir(...)`.
- Al ser `open` permite que subclases sobrescriban comportamiento si fuera necesario.

### Subclase: `CalculadoraCientifica : Calculadora()`
Hereda toda la funcionalidad de `Calculadora` (incluida la memoria) y la extiende con:
- Funciones trigonométricas (seno, coseno, tangente y sus inversas).
- Potencias, raíces, logaritmos, exponenciales.
- Factorial con límites y porcentaje.
- Control de modo (radianes/grados).


---

## Polimorfismo 

### A. Polimorfismo de subtipado (herencia + interfaces)
`Memorizable` es implementada por `Calculadora`.  
Cualquier referencia tipada como `Memorizable` puede apuntar a una `Calculadora` o a una subclase (`CalculadoraCientifica`).

**Ejemplo:**
```kotlin
val m: Memorizable = CalculadoraCientifica()
m.memoryAdd(5.0) // funciona sin conocer la implementación exacta
