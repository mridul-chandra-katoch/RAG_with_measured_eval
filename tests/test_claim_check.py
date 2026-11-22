import unittest
from ragkit.claim_check import check_claims
from ragkit.schemas import Citation, Chunk

class TestClaimCheck(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk("c1", "d1", "Attendance requires 75 percent of all scheduled classes to pass the course.", {"file": "test.md", "page": 1}),
            Chunk("c2", "d1", "The sky is blue in the document we read yesterday.", {"file": "test.md", "page": 1}),
        ]
        self.citations = [
            Citation("c1", "test.md", 1, "quote"),
            Citation("c2", "test.md", 1, "quote"),
        ]

    def test_exact_match(self):
        # This is a literal substring
        ans = "Attendance requires 75 percent of all scheduled classes."
        res = check_claims(ans, self.citations, self.chunks)
        self.assertEqual(res["overall_label"], "cited")
        self.assertEqual(res["sentences"][0]["label"], "cited")

    def test_hallucination(self):
        # This is NOT in the text (changed 75 to 80)
        ans = "Attendance requires 80 percent of all scheduled classes."
        res = check_claims(ans, self.citations, self.chunks)
        self.assertEqual(res["overall_label"], "unverified")
        self.assertEqual(res["sentences"][0]["label"], "unverified")

    def test_multi_sentence_one_fails(self):
        ans = "Attendance requires 75 percent of all scheduled classes. The sky is green."
        res = check_claims(ans, self.citations, self.chunks)
        # Overall should be unverified because the second sentence is hallucinated
        self.assertEqual(res["overall_label"], "unverified")
        self.assertEqual(res["sentences"][0]["label"], "cited")
        self.assertEqual(res["sentences"][1]["label"], "unverified")

    def test_short_sentence(self):
        # Less than 6 words, cannot be cited
        ans = "The sky is blue."
        res = check_claims(ans, self.citations, self.chunks)
        self.assertEqual(res["overall_label"], "unverified")

if __name__ == "__main__":
    unittest.main()