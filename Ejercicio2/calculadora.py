from mesa import Agent, Model
from mesa.time import BaseScheduler
import math
import re
import uuid


# -------------------------
# Operación: agente calculador
# -------------------------
class OperationAgent(Agent):
    def __init__(self, unique_id, model, op_name):
        super().__init__(unique_id, model)
        self.op_name = op_name  # 'add', 'sub', 'mul', 'div', 'pow'
        self.inbox = []

    def step(self):
        while self.inbox:
            msg = self.inbox.pop(0)
            if msg.get("type") == "task":
                left = msg.get("left")
                right = msg.get("right")
                task_id = msg.get("task_id")
                reply_to = msg.get("reply_to")
                if not (isinstance(left, (int, float)) and isinstance(right, (int, float))):
                    err_msg = {
                        "type": "error",
                        "task_id": task_id,
                        "from": self.unique_id,
                        "error": "Operandos no numéricos recibidos por agent {}".format(self.op_name),
                    }
                    self.model.send_message(err_msg, reply_to)
                    continue
                try:
                    val = self.compute(left, right)
                except Exception as e:
                    err_msg = {
                        "type": "error",
                        "task_id": task_id,
                        "from": self.unique_id,
                        "error": str(e),
                    }
                    self.model.send_message(err_msg, reply_to)
                    continue
                res_msg = {
                    "type": "result",
                    "task_id": task_id,
                    "value": val,
                    "from": self.unique_id,
                }
                self.model.send_message(res_msg, reply_to)

    def compute(self, left, right):
        if self.op_name == "add":
            return left + right
        elif self.op_name == "sub":
            return left - right
        elif self.op_name == "mul":
            return left * right
        elif self.op_name == "div":
            if right == 0:
                raise ZeroDivisionError("División por cero en task")
            return left / right
        elif self.op_name == "pow":
            return math.pow(left, right)
        else:
            raise ValueError("Operación desconocida: {}".format(self.op_name))


# -------------------------
# IO Agent
# -------------------------
class IOAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.inbox = []
        self.tasks = {}
        self.pending = set()
        self.awaiting = {}
        self.root_task_id = None
        self.finished = False
        self.last_result = None

    def submit_expression(self, expression):
        self.tasks.clear()
        self.pending.clear()
        self.awaiting.clear()
        self.root_task_id = None
        self.finished = False
        self.last_result = None

        tasks, root = parse_expression_to_tasks(expression)
        for tid, t in tasks.items():
            op = t["op"]
            if op == "+":
                agent_id = "op_add"
            elif op == "-":
                agent_id = "op_sub"
            elif op == "*":
                agent_id = "op_mul"
            elif op == "/":
                agent_id = "op_div"
            elif op == "^":
                agent_id = "op_pow"
            elif op == "const":
                agent_id = None
            else:
                raise ValueError("Operador desconocido en tarea: {}".format(op))
            t["assigned_agent"] = agent_id
            self.tasks[tid] = t
            self.pending.add(tid)
        self.root_task_id = root

        for tid, t in tasks.items():
            for operand in (t["left"], t["right"]):
                if isinstance(operand, str) and operand in self.tasks:
                    self.awaiting.setdefault(operand, []).append(tid)

        self.dispatch_ready_tasks()

    def step(self):
        while self.inbox:
            msg = self.inbox.pop(0)
            mtype = msg.get("type")
            if mtype == "result":
                task_id = msg.get("task_id")
                value = msg.get("value")
                self.handle_result(task_id, value)
            elif mtype == "error":
                # Registrar error y marcar fin
                # Guardamos mensaje de error en last_result como cadena para mostrar al usuario
                self.last_result = "ERROR: " + str(msg.get("error"))
                self.finished = True
                print(self.last_result)
            else:
                print("IO: mensaje desconocido recibido:", msg)

    def handle_result(self, task_id, value):
        dependents = self.awaiting.get(task_id, [])
        for dep_tid in dependents:
            dep_task = self.tasks.get(dep_tid)
            if dep_task is None:
                continue
            if dep_task["left"] == task_id:
                dep_task["left"] = value
            if dep_task["right"] == task_id:
                dep_task["right"] = value
        if task_id == self.root_task_id:
            self.last_result = value
            self.finished = True
            print("Resultado final:", value)
            return
        self.pending.discard(task_id)
        if task_id in self.awaiting:
            del self.awaiting[task_id]
        self.dispatch_ready_tasks()

    def dispatch_ready_tasks(self):
        to_send = []
        for tid in list(self.pending):
            t = self.tasks[tid]
            left = t["left"]
            right = t["right"]
            if t["op"] == "const":
                # auto-resolver constantes
                self.handle_result(tid, left)
                self.pending.discard(tid)
                continue
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                to_send.append(tid)
        for tid in to_send:
            t = self.tasks[tid]
            agent_id = t["assigned_agent"]
            if agent_id is None:
                continue
            msg = {
                "type": "task",
                "task_id": tid,
                "left": t["left"],
                "right": t["right"],
                "reply_to": self.unique_id,
            }
            self.model.send_message(msg, agent_id)

    def receive_message(self, msg):
        self.inbox.append(msg)


# -------------------------
# Modelo
# -------------------------
class CalculatorModel(Model):
    def __init__(self):
        super().__init__()
        self.schedule = BaseScheduler(self)
        ops = [
            ("op_add", "add"),
            ("op_sub", "sub"),
            ("op_mul", "mul"),
            ("op_div", "div"),
            ("op_pow", "pow"),
        ]
        self.agents = {}
        for aid, opname in ops:
            ag = OperationAgent(aid, self, opname)
            self.agents[aid] = ag
            self.schedule.add(ag)
        io = IOAgent("io", self)
        self.io_agent = io
        self.agents["io"] = io
        self.schedule.add(io)
        self.messages = []

    def send_message(self, msg, recipient_id):
        self.messages.append((recipient_id, msg))

    def deliver_messages(self):
        if not self.messages:
            return
        to_deliver = self.messages.copy()
        self.messages.clear()
        for recipient_id, msg in to_deliver:
            if recipient_id not in self.agents:
                print("Advertencia: receptor desconocido '{}' para mensaje {}".format(recipient_id, msg))
                continue
            agent = self.agents[recipient_id]
            if isinstance(agent, IOAgent):
                agent.receive_message(msg)
            else:
                agent.inbox.append(msg)

    def step(self):
        # Delivery antes y después para mantener comunicación síncrona
        self.deliver_messages()
        self.schedule.step()
        self.deliver_messages()


# -------------------------
# Parser & Task builder
# -------------------------
token_specification = [
    ("NUMBER", r"\d+(\.\d+)?"),  # Integer or decimal
    ("OP", r"[\+\-\*\/\^]"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("SKIP", r"[ \t]+"),
]
token_regex = "|".join("(?P<%s>%s)" % pair for pair in token_specification)
get_token = re.compile(token_regex).match


def tokenize(expr):
    tokens = []
    pos = 0
    prev_tok = None
    length = len(expr)
    while pos < length:
        m = get_token(expr, pos)
        if not m:
            raise SyntaxError("Token inválido en posición %d: '%s'" % (pos, expr[pos:]))
        typ = m.lastgroup
        tok = m.group(typ)
        pos = m.end()
        if typ == "SKIP":
            continue
        if typ == "OP" and tok == "-":
            if prev_tok is None or prev_tok in ("OP", "LPAREN"):
                tokens.append("0")
                tokens.append("-")
                prev_tok = "OP"
                continue
        tokens.append(tok)
        prev_tok = typ if typ != "SKIP" else prev_tok
    return tokens


precedence = {
    "+": 2,
    "-": 2,
    "*": 3,
    "/": 3,
    "^": 4,
}
right_associative = {"^"}


def shunting_yard(tokens):
    output = []
    stack = []
    for tok in tokens:
        if re.fullmatch(r"\d+(\.\d+)?", tok):
            output.append(tok)
        elif tok in precedence:
            while stack and stack[-1] in precedence:
                top = stack[-1]
                if (
                    (top not in right_associative and precedence[top] >= precedence[tok])
                    or (top in right_associative and precedence[top] > precedence[tok])
                ):
                    output.append(stack.pop())
                else:
                    break
            stack.append(tok)
        elif tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if not stack:
                raise SyntaxError("Paréntesis sin pareja")
            stack.pop()
        else:
            raise SyntaxError("Token desconocido en shunting_yard: {}".format(tok))
    while stack:
        if stack[-1] in ("(", ")"):
            raise SyntaxError("Paréntesis sin pareja al finalizar")
        output.append(stack.pop())
    return output


def build_tasks_from_rpn(rpn):
    stack = []
    tasks = {}
    for tok in rpn:
        if re.fullmatch(r"\d+(\.\d+)?", tok):
            if "." in tok:
                val = float(tok)
            else:
                val = int(tok)
            stack.append(val)
        elif tok in precedence:
            if len(stack) < 2:
                raise SyntaxError("Faltan operandos para operador '{}'".format(tok))
            right = stack.pop()
            left = stack.pop()
            tid = "t_{}".format(uuid.uuid4().hex[:8])
            tasks[tid] = {"op": tok, "left": left, "right": right}
            stack.append(tid)
        else:
            raise SyntaxError("Token inesperado en build_tasks_from_rpn: {}".format(tok))
    if len(stack) != 1:
        raise SyntaxError("Expresión inválida - pila final != 1")
    root = stack[0]
    if isinstance(root, (int, float)):
        tid = "t_root_const_{}".format(uuid.uuid4().hex[:6])
        tasks[tid] = {"op": "const", "left": root, "right": 0}
        return tasks, tid
    return tasks, root


def parse_expression_to_tasks(expression):
    tokens = tokenize(expression)
    rpn = shunting_yard(tokens)
    tasks, root = build_tasks_from_rpn(rpn)
    return tasks, root


# -------------------------
# Ejecución / uso interactivo
# -------------------------
def run_expression(expr, max_steps=100):
    """
    Ejecuta el modelo hasta que IOAgent termine o hasta max_steps.
    Retorna el resultado (número o string de error) o None si timeout.
    """
    model = CalculatorModel()
    io = model.io_agent
    # No imprimir "Evaluando" aquí para que el prompt sea más limpio; el IO imprimirá el resultado.
    io.submit_expression(expr)
    steps = 0
    while not io.finished and steps < max_steps:
        model.step()
        steps += 1
    if not io.finished:
        return None
    return io.last_result


def interactive_loop():
    print("Calculadora basada en agentes (escriba 'salir' para terminar).")
    try:
        while True:
            try:
                expr = input("Ingresa expresión: ").strip()
            except EOFError:
                print("\nFin de entrada. Saliendo.")
                break
            if expr.lower() in ("salir", "exit", "quit"):
                print("Saliendo.")
                break
            if expr == "":
                continue
            try:
                result = run_expression(expr, max_steps=500)
                if result is None:
                    print("No se obtuvo resultado (timeout o error interno).")
                else:
                    print("=>", result)
            except SyntaxError as e:
                print("Error de sintaxis:", e)
            except Exception as e:
                print("Error ejecutando la expresión:", e)
    except KeyboardInterrupt:
        print("\nInterrupción por teclado. Saliendo.")


if __name__ == "__main__":
    interactive_loop()