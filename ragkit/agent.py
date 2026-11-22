import re
import sqlite3
import ast
from ragkit.schemas import Answer

ALLOWED_SQL_TABLES = {"students"}
DB_PATH = "results/agent.db"

def _init_sql_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS students (id INTEGER, name TEXT, grade INTEGER)')
    c.execute('DELETE FROM students')
    c.executemany('INSERT INTO students VALUES (?, ?, ?)', [(1, 'Alice', 85), (2, 'Bob', 92), (3, 'Charlie', 78)])
    conn.commit()
    conn.close()

def calculate(expression):
    safe_expr = re.sub(r'[^0-9+\-*/().]', '', expression)
    if not safe_expr: return "Invalid expression"
    try:
        tree = ast.parse(safe_expr, mode='eval')
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)):
                return "Error: Unsupported operation"
        return str(eval(compile(tree, '<string>', 'eval'), {"__builtins__": None}, {}))
    except Exception:
        return "Error: Invalid syntax"

def query_sql(question):
    _init_sql_db()
    q_lower = question.lower()
    sql = ""
    if 'highest' in q_lower or 'max' in q_lower:
        sql = "SELECT name, grade FROM students ORDER BY grade DESC LIMIT 1"
    elif 'lowest' in q_lower or 'min' in q_lower:
        sql = "SELECT name, grade FROM students ORDER BY grade ASC LIMIT 1"
    elif 'how many' in q_lower or 'count' in q_lower:
        sql = "SELECT COUNT(*) FROM students"
    else:
        sql = "SELECT name, grade FROM students"
        
    # Schema allowlist check
    if not any(table in sql for table in ALLOWED_SQL_TABLES):
        return "Error: Unauthorized table access."
        
    try:
        # REAL read-only connection
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        c = conn.cursor()
        c.execute(sql)
        res = c.fetchall()
        conn.close()
        
        if not res: return "No results."
        if len(res[0]) == 1: return f"Result: {res[0][0]}"
        if len(res) == 1: return f"Result: {res[0][0]} with {res[0][1]}."
        return "Results:\n" + "\n".join(f"- {r[0]}: {r[1]}" for r in res)
    except Exception as e:
        return "SQL Error."

def run_agent(question, pipeline):
    q_lower = question.lower().strip()
    
    if q_lower.startswith("calculate "):
        return Answer(calculate(question), False, [], 1.0, "tool"), "Calculator Tool"
        
    if q_lower.startswith("sql: "):
        return Answer(query_sql(question), False, [], 1.0, "tool"), "SQL Tool"
        
    ans = pipeline.ask(question)
    return ans, "Document Search Tool"