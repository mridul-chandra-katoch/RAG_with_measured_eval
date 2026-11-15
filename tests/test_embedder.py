import unittest
from ragkit.embedder import HashingEmbedder

class TestEmbedder(unittest.TestCase):
    def setUp(self):
        self.embedder = HashingEmbedder(dim=128)

    def test_dimensions(self):
        vec = self.embedder.embed("Hello world")
        self.assertEqual(len(vec), 128)

    def test_l2_normalization(self):
        vec = self.embedder.embed("Hello world")
        norm = sum(v * v for v in vec)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_deterministic(self):
        vec1 = self.embedder.embed("Hello world")
        vec2 = self.embedder.embed("Hello world")
        self.assertEqual(vec1, vec2)

    def test_empty(self):
        vec = self.embedder.embed("")
        self.assertEqual(sum(vec), 0.0)

if __name__ == "__main__":
    unittest.main()