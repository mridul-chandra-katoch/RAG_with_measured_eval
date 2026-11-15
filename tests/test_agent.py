import unittest
from ragkit.agent import run_agent
from ragkit.schemas import Answer

class MockPipeline:
    def ask(self, q):
        return Answer("Document Answer", False, [], 1.0, "123")

class TestAgentRouter(unittest.TestCase):
    def setUp(self):
        self.pipeline = MockPipeline()

    def test_student_question_routes_to_document(self):
        ans, tool = run_agent("What are the grading rules for students?", self.pipeline)
        self.assertEqual(tool, "Document Search Tool")

    def test_fee_question_routes_to_document(self):
        ans, tool = run_agent("What is the 2024-25 fee?", self.pipeline)
        self.assertEqual(tool, "Document Search Tool")

    def test_calculator_routes_correctly(self):
        ans, tool = run_agent("calculate 2 + 2", self.pipeline)
        self.assertEqual(tool, "Calculator Tool")
        self.assertEqual(ans.answer, "4")
        
    def test_sql_routes_correctly(self):
        ans, tool = run_agent("sql: select * from students", self.pipeline)
        self.assertEqual(tool, "SQL Tool")

if __name__ == "__main__":
    unittest.main()