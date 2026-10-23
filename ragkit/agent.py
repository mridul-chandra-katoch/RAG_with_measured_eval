import re
import sqlite3
from ragkit.schemas import Answer

def calculate(expression):
    safe_expr = re.sub(r'[^0-9+\-*/().]', '', expression)
    if not safe_expr:
        return "Invalid expression"
    try:
        return str(eval(safe_expr, {"__builtins__": None}, {}))
    except Exception as e:
        return "Error: " + str(e)

def query_sql(question):
    conn = sqlite3.connect(':memory:')
    c = conn.cursor()
    c.execute('CREATE TABLE students (id INTEGER, name TEXT, grade INTEGER)')
    c.executemany('INSERT INTO students VALUES (?, ?, ?)', [
        (1, 'Alice', 85), (2, 'Bob', 92), (3, 'Charlie', 78)
    ])
    conn.commit()
    
    q_lower = question.lower()
    if 'highest' in q_lower or 'max' in q_lower:
        c.execute('SELECT name, grade FROM students ORDER BY grade DESC LIMIT 1')
        res = c.fetchone()
        return f"The student with the highest grade is {res[0]} with {res[1]}."
    elif 'lowest' in q_lower or 'min' in q_lower:
        c.execute('SELECT name, grade FROM students ORDER BY grade ASC LIMIT 1')
        res = c.fetchone()
        return f"The student with the lowest grade is {res[0]} with {res[1]}."
    elif 'how many' in q_lower or 'count' in q_lower:
        c.execute('SELECT COUNT(*) FROM students')
        res = c.fetchone()
        return f"There are {res[0]} students in the database."
    else:
        c.execute('SELECT name, grade FROM students')
        res = c.fetchall()
        out = "Student Records:\n"
        for r in res:
            out += f"- {r[0]}: {r[1]}\n"
        return out

def run_agent(question, pipeline):
    q_lower = question.lower()
    
    if 'calculate' in q_lower or 'math' in q_lower or re.search(r'\d+[\+\-\*/]', q_lower):
        return Answer(calculate(question), False, [], 1.0, "tool"), "Calculator Tool"
        
    if 'student' in q_lower or 'database' in q_lower or 'grade' in q_lower:
        return Answer(query_sql(question), False, [], 1.0, "tool"), "SQL Tool"
        
    ans = pipeline.ask(question)
    return ans, "Document Search Tool"