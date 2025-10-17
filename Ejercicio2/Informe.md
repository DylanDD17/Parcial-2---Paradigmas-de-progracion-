
# Calculadora basada en agentes





## Componentes clave
- CalculatorModel: cola global de mensajes, entrega síncrona (deliver_messages), scheduler.
- IOAgent: parseo (tokenize → shunting‑yard → RPN), generación de tareas, tablas:
  - tasks (task_id → {op,left,right,assigned_agent})
  - pending (tareas pendientes)
  - awaiting (mapa de dependencias)
  - root_task_id, last_result, finished.
- OperationAgent (op_add, op_sub, op_mul, op_div, op_pow): reciben 'task' con operandos numéricos, devuelven 'result' o 'error'.

## Formato de mensajes
- Task (IO → OperationAgent):
  - { type: "task", task_id, left, right, reply_to }
- Result (OperationAgent → IO):
  - { type: "result", task_id, value, from }
- Error:
  - { type: "error", task_id, error, from }

## Flujo de evaluación (esencial)
1. Usuario envía expresión → IOAgent.parse → RPN → build_tasks → tasks DAG.
2. IO identifica tareas "listas" (ambos operandos numéricos) y envía mensajes 'task' a los agentes correspondientes.
3. OperationAgent calcula y envía 'result' al reply_to (IO).
4. IO recibe 'result', sustituye task_id por valor en tareas dependientes, actualiza awaiting/pending.
5. IO envía nuevas tareas listas hasta que se resuelve root_task_id → resultado final devuelto.

## Sincronización y entrega de mensajes
- Mensajes se encolan con model.send_message y se entregan en deliver_messages.
- Ciclo por tick: deliver_messages → schedule.step() (agents.step) → deliver_messages.
- Garantiza orden determinista en ejecución single‑process / single‑thread.

## Manejo de errores (esencial)
- Division por cero → OperationAgent envía 'error' → IO marca finished y reporta error.
- Errores de parsing (tokenización/shunting) → excepción manejada en el bucle interactivo.
- IO evita enviar tareas con operandos no numéricos; si ocurre, OperationAgent devuelve 'error'.

