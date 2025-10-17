import kotlin.math.*

// --- Excepciones ---
class CalculatorException(message: String) : Exception(message)

// --- Interfaz de memoria ---
interface Memorizable {
    fun memoryAdd(value: Double)
    fun memorySubtract(value: Double)
    fun memoryRecall(): Double
    fun memoryClear()
}

// --- Calculadora básica ---
open class Calculadora : Memorizable {
    private var memory: Double = 0.0

    open fun sumar(a: Number, b: Number): Double = a.toDouble() + b.toDouble()
    open fun restar(a: Number, b: Number): Double = a.toDouble() - b.toDouble()
    open fun multiplicar(a: Number, b: Number): Double = a.toDouble() * b.toDouble()
    open fun dividir(a: Number, b: Number): Double {
        if (b.toDouble() == 0.0) throw CalculatorException("Error: División por cero")
        return a.toDouble() / b.toDouble()
    }

    // Memoria
    override fun memoryAdd(value: Double) { memory += value }
    override fun memorySubtract(value: Double) { memory -= value }
    override fun memoryRecall(): Double = memory
    override fun memoryClear() { memory = 0.0 }
}

// --- Calculadora científica ---
open class CalculadoraCientifica : Calculadora() {
    var modoRadianes: Boolean = true
    var ans: Double = 0.0

    private fun toRadiansIfNeeded(x: Double): Double = if (modoRadianes) x else Math.toRadians(x)
    private fun fromRadiansIfNeeded(x: Double): Double = if (modoRadianes) x else Math.toDegrees(x)

    fun seno(x: Double): Double = sin(toRadiansIfNeeded(x))
    fun coseno(x: Double): Double = cos(toRadiansIfNeeded(x))
    fun tangente(x: Double): Double = tan(toRadiansIfNeeded(x))
    fun arcSeno(x: Double): Double = fromRadiansIfNeeded(asin(x))
    fun arcCoseno(x: Double): Double = fromRadiansIfNeeded(acos(x))
    fun arcTangente(x: Double): Double = fromRadiansIfNeeded(atan(x))

    fun potencia(base: Double, exponente: Double): Double = base.pow(exponente)
    fun raiz(x: Double, n: Double): Double {
        if (n == 0.0) throw CalculatorException("Error: Índice de raíz no puede ser 0")
        return x.pow(1.0 / n)
    }
    fun raizCuadrada(x: Double): Double {
        if (x < 0) throw CalculatorException("Error: Raíz de número negativo")
        return sqrt(x)
    }
    fun logaritmoBase10(x: Double): Double {
        if (x <= 0) throw CalculatorException("Error: Logaritmo de número no positivo")
        return log10(x)
    }
    fun logaritmoNeperiano(x: Double): Double {
        if (x <= 0) throw CalculatorException("Error: Logaritmo de número no positivo")
        return ln(x)
    }
    fun exponencial(x: Double): Double = exp(x)

    // Factorial iterativo, límite para evitar overflow y consumo excesivo
    fun factorialInt(n: Int): Long {
        if (n < 0) throw CalculatorException("Error: Factorial de número negativo")
        if (n > 20) throw CalculatorException("Error: Factorial demasiado grande (límite 20)")
        var res = 1L
        for (i in 2..n) res *= i
        return res
    }

    fun porcentaje(xPercent: Double): Double = xPercent / 100.0

    fun cambiarModo(radianes: Boolean) { modoRadianes = radianes }

    val PI: Double get() = Math.PI
    val E: Double get() = Math.E
}

// --- Evaluador de expresiones (shunting-yard -> RPN) ---
class ExpressionEvaluator(private val calc: CalculadoraCientifica) {
    companion object {
        private val FUNCTIONS = setOf("sin","cos","tan","asin","acos","atan","log","ln","sqrt","exp")
        private val OPERATORS = setOf("+","-","*","/","^")
        private val PRECEDENCE = mapOf(
            "+" to 2,
            "-" to 2,
            "*" to 3,
            "/" to 3,
            "^" to 4,
            "u-" to 5 // unary minus
        )
        private const val MAX_EXPRESSION_LENGTH = 1000 // evita expresiones gigantes
    }

    private sealed class Token {
        data class Number(val value: Double): Token()
        data class Operator(val op: String): Token()
        data class Function(val name: String): Token()
        object LeftParen: Token()
        object RightParen: Token()
        data class Constant(val name: String): Token()
        data class Postfix(val sym: String): Token() // '!' or '%'
    }

    fun evaluate(expr: String): Double {
        val e = expr.trim()
        if (e.isEmpty()) throw CalculatorException("Expresión vacía")
        if (e.length > MAX_EXPRESSION_LENGTH) throw CalculatorException("Expresión demasiado larga")
        val tokens = tokenize(e)
        val rpn = toRPN(tokens)
        val result = evalRPN(rpn)
        calc.ans = result
        return result
    }

    private fun tokenize(s: String): List<Token> {
        val tokens = mutableListOf<Token>()
        var i = 0
        fun peek(): Char? = if (i < s.length) s[i] else null

        while (i < s.length) {
            val ch = s[i]
            when {
                ch.isWhitespace() -> i++
                ch.isDigit() || (ch == '.' && peek()?.let { it.isDigit() } == true) -> {
                    val start = i
                    while (i < s.length && (s[i].isDigit() || s[i] == '.')) i++
                    val numStr = s.substring(start, i)
                    val num = numStr.toDoubleOrNull() ?: throw CalculatorException("Número inválido: $numStr")
                    tokens.add(Token.Number(num))
                }
                ch.isLetter() -> {
                    val start = i
                    while (i < s.length && (s[i].isLetter())) i++
                    val name = s.substring(start, i).lowercase()
                    when (name) {
                        "pi", "π" -> tokens.add(Token.Number(calc.PI))
                        "e" -> tokens.add(Token.Number(calc.E))
                        "ans" -> tokens.add(Token.Number(calc.ans))
                        else -> {
                            if (FUNCTIONS.contains(name)) tokens.add(Token.Function(name))
                            else throw CalculatorException("Función o constante desconocida: $name")
                        }
                    }
                }
                ch == ',' -> { // separador de argumentos (si se implementaran funciones multi-arg)
                    i++
                    // handled in shunting-yard as argument separator if ever needed
                }
                ch == '(' -> { tokens.add(Token.LeftParen); i++ }
                ch == ')' -> { tokens.add(Token.RightParen); i++ }
                ch == '!' -> { tokens.add(Token.Postfix("!")); i++ }
                ch == '%' -> { tokens.add(Token.Postfix("%")); i++ }
                ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '^' -> {
                    // Detect unary minus: if '-' is at start or after left paren or after operator
                    if (ch == '-') {
                        val prev = tokens.lastOrNull()
                        if (prev == null || prev is Token.Operator || prev is Token.LeftParen || prev is Token.Function) {
                            tokens.add(Token.Operator("u-"))
                            i++
                            continue
                        }
                    }
                    tokens.add(Token.Operator(ch.toString()))
                    i++
                }
                else -> throw CalculatorException("Carácter inválido en la expresión: '$ch'")
            }
        }
        return tokens
    }

    private fun toRPN(tokens: List<Token>): List<Token> {
        val out = mutableListOf<Token>()
        val stack = java.util.ArrayDeque<Token>()
        for (token in tokens) {
            when (token) {
                is Token.Number -> out.add(token)
                is Token.Constant -> out.add(token)
                is Token.Function -> stack.push(token)
                is Token.Operator -> {
                    val op1 = token.op
                    while (stack.isNotEmpty()) {
                        val top = stack.peek()
                        if (top is Token.Operator) {
                            val op2 = top.op
                            val p1 = PRECEDENCE[op1] ?: 0
                            val p2 = PRECEDENCE[op2] ?: 0
                            // '^' is right-assoc; unary has highest precedence already set
                            val rightAssoc = (op1 == "^")
                            if ((rightAssoc && p1 < p2) || (!rightAssoc && p1 <= p2)) {
                                out.add(stack.pop())
                                continue
                            }
                        } else if (top is Token.Function) {
                            out.add(stack.pop())
                            continue
                        }
                        break
                    }
                    stack.push(token)
                }
                is Token.LeftParen -> stack.push(token)
                is Token.RightParen -> {
                    while (stack.isNotEmpty() && stack.peek() !is Token.LeftParen) {
                        out.add(stack.pop())
                    }
                    if (stack.isEmpty() || stack.peek() !is Token.LeftParen) {
                        throw CalculatorException("Paréntesis desbalanceados")
                    }
                    stack.pop() // pop left paren
                    if (stack.isNotEmpty() && stack.peek() is Token.Function) {
                        out.add(stack.pop())
                    }
                }
                is Token.Postfix -> {
                    // Postfix operators go immediately to output (they apply to previous output token)
                    out.add(token)
                }
            }
        }
        while (stack.isNotEmpty()) {
            val top = stack.pop()
            if (top is Token.LeftParen || top is Token.RightParen) throw CalculatorException("Paréntesis desbalanceados")
            out.add(top)
        }
        return out
    }

    private fun evalRPN(rpn: List<Token>): Double {
        val st = java.util.ArrayDeque<Double>()
        for (t in rpn) {
            when (t) {
                is Token.Number -> st.push(t.value)
                is Token.Operator -> {
                    when (t.op) {
                        "u-" -> {
                            val v = st.pollFirst() ?: throw CalculatorException("Operando faltante para unary -")
                            st.push(-v)
                        }
                        "+" -> {
                            val b = st.pollFirst() ?: throw CalculatorException("Operando faltante")
                            val a = st.pollFirst() ?: throw CalculatorException("Operando faltante")
                            st.push(a + b)
                        }
                        "-" -> {
                            val b = st.pollFirst() ?: throw CalculatorException("Operando faltante")
                            val a = st.pollFirst() ?: throw CalculatorException("Operando faltante")
                            st.push(a - b)
                        }
                        "*" -> {
                            val b = st.pollFirst() ?: throw CalculatorException("Operando faltante")
                            val a = st.pollFirst() ?: throw CalculatorException("Operando faltante")
                            st.push(a * b)
                        }
                        "/" -> {
                            val b = st.pollFirst() ?: throw CalculatorException("Operando faltante")
                            val a = st.pollFirst() ?: throw CalculatorException("Operando faltante")
                            if (b == 0.0) throw CalculatorException("Error: División por cero")
                            st.push(a / b)
                        }
                        "^" -> {
                            val ex = st.pollFirst() ?: throw CalculatorException("Operando faltante")
                            val base = st.pollFirst() ?: throw CalculatorException("Operando faltante")
                            st.push(base.pow(ex))
                        }
                        else -> throw CalculatorException("Operador desconocido: ${t.op}")
                    }
                }
                is Token.Function -> {
                    when (t.name) {
                        "sin" -> {
                            val v = st.pollFirst() ?: throw CalculatorException("Argumento faltante para sin")
                            st.push(calc.seno(v))
                        }
                        "cos" -> {
                            val v = st.pollFirst() ?: throw CalculatorException("Argumento faltante para cos")
                            st.push(calc.coseno(v))
                        }
                        "tan" -> {
                            val v = st.pollFirst() ?: throw CalculatorException("Argumento faltante para tan")
                            st.push(calc.tangente(v))
                        }
                        "asin" -> {
                            val v = st.pollFirst() ?: throw CalculatorException("Argumento faltante para asin")
                            st.push(calc.arcSeno(v))
                        }
                        "acos" -> {
                            val v = st.pollFirst() ?: throw CalculatorException("Argumento faltante para acos")
                            st.push(calc.arcCoseno(v))
                        }
                        "atan" -> {
                            val v = st.pollFirst() ?: throw CalculatorException("Argumento faltante para atan")
                            st.push(calc.arcTangente(v))
                        }
                        "log" -> {
                            val v = st.pollFirst() ?: throw CalculatorException("Argumento faltante para log")
                            st.push(calc.logaritmoBase10(v))
                        }
                        "ln" -> {
                            val v = st.pollFirst() ?: throw CalculatorException("Argumento faltante para ln")
                            st.push(calc.logaritmoNeperiano(v))
                        }
                        "sqrt" -> {
                            val v = st.pollFirst() ?: throw CalculatorException("Argumento faltante para sqrt")
                            st.push(calc.raizCuadrada(v))
                        }
                        "exp" -> {
                            val v = st.pollFirst() ?: throw CalculatorException("Argumento faltante para exp")
                            st.push(calc.exponencial(v))
                        }
                        else -> throw CalculatorException("Función desconocida: ${t.name}")
                    }
                }
                is Token.Postfix -> {
                    when (t.sym) {
                        "!" -> {
                            val v = st.pollFirst() ?: throw CalculatorException("Operando faltante para !")
                            val iv = v.toInt()
                            if (iv.toDouble() != v) throw CalculatorException("Factorial requiere entero")
                            val res = calc.factorialInt(iv)
                            st.push(res.toDouble())
                        }
                        "%" -> {
                            val v = st.pollFirst() ?: throw CalculatorException("Operando faltante para %")
                            st.push(calc.porcentaje(v))
                        }
                        else -> throw CalculatorException("Postfix desconocido: ${t.sym}")
                    }
                }
                else -> throw CalculatorException("Token no manejado en evalRPN")
            }
        }
        if (st.size != 1) throw CalculatorException("Expresión inválida (número de operandos incorrecto)")
        return st.pop()
    }
}

// --- Menú principal y uso ---
fun main() {
    val calc = CalculadoraCientifica()
    val evaluator = ExpressionEvaluator(calc)
    var continuar = true
    println("=== Calculadora Científica (con evaluador seguro) ===")
    while (continuar) {
        println("\nSeleccione una opción:")
        println("1. Operación básica (+, -, *, /)")
        println("2. Operación científica simple (sin, cos, tan, log, ln, sqrt, ^, !)")
        println("3. Evaluar expresión completa (ej: 2 + 3 * sin(45) - log(10))")
        println("4. Memoria (M+ valor, M- valor, MR, MC)")
        println("5. Cambiar modo (Grados/Radianes)")
        println("0. Salir")

        when (readLine()?.trim()) {
            "1" -> {
                println("Ingrese operación: ejemplo: 5 + 3")
                val input = readLine() ?: ""
                val parts = input.split(" ")
                if (parts.size == 3) {
                    val a = parts[0].toDoubleOrNull()
                    val op = parts[1]
                    val b = parts[2].toDoubleOrNull()
                    if (a != null && b != null) {
                        val res = try {
                            when (op) {
                                "+" -> calc.sumar(a, b)
                                "-" -> calc.restar(a, b)
                                "*" -> calc.multiplicar(a, b)
                                "/" -> calc.dividir(a, b)
                                else -> throw CalculatorException("Operador no válido")
                            }
                        } catch (e: CalculatorException) { println("Resultado: ${e.message}"); null }
                        if (res != null) println("Resultado: $res")
                    } else println("Entradas no válidas")
                } else println("Formato incorrecto")
            }
            "2" -> {
                println("Ingrese operación científica: ejemplo 'sin 45' o 'sqrt 9' o '! 5' (nota: para potencia use opción 3 o '2^3')")
                val input = readLine() ?: ""
                val parts = input.trim().split(Regex("\\s+"))
                try {
                    val res = when (parts[0].lowercase()) {
                        "sin" -> calc.seno(parts[1].toDouble())
                        "cos" -> calc.coseno(parts[1].toDouble())
                        "tan" -> calc.tangente(parts[1].toDouble())
                        "log" -> calc.logaritmoBase10(parts[1].toDouble())
                        "ln" -> calc.logaritmoNeperiano(parts[1].toDouble())
                        "sqrt" -> calc.raizCuadrada(parts[1].toDouble())
                        "!" -> calc.factorialInt(parts[1].toInt()).toDouble()
                        else -> {
                            println("Operación no válida. Use la opción 3 para expresiones completas.")
                            null
                        }
                    }
                    if (res != null) println("Resultado: $res")
                } catch (e: Exception) {
                    println("Error: ${e.message}")
                }
            }
            "3" -> {
                println("Ingrese la expresión (ejemplo: 2 + 3 * sin(45) - log(10)):")
                val expr = readLine() ?: ""
                try {
                    val res = evaluator.evaluate(expr)
                    println("Resultado: $res")
                } catch (e: Exception) {
                    println("Error: ${e.message}")
                }
            }
            "4" -> {
                println("Opciones de memoria: M+ valor, M- valor, MR, MC")
                val memInput = readLine() ?: ""
                val parts = memInput.trim().split(Regex("\\s+"))
                when (parts[0].uppercase()) {
                    "M+" -> calc.memoryAdd(parts.getOrNull(1)?.toDoubleOrNull() ?: 0.0)
                    "M-" -> calc.memorySubtract(parts.getOrNull(1)?.toDoubleOrNull() ?: 0.0)
                    "MR" -> println("Memoria: ${calc.memoryRecall()}")
                    "MC" -> { calc.memoryClear(); println("Memoria borrada") }
                    else -> println("Comando de memoria no válido")
                }
            }
            "5" -> {
                println("Modo actual: ${if (calc.modoRadianes) "Radianes" else "Grados"}")
                println("¿Desea cambiar? (s/n)")
                if ((readLine() ?: "").lowercase() == "s") {
                    calc.cambiarModo(!calc.modoRadianes)
                    println("Nuevo modo: ${if (calc.modoRadianes) "Radianes" else "Grados"}")
                }
            }
            "0" -> continuar = false
            else -> println("Opción no válida")
        }
    }
    println("¡Hasta luego!")
}