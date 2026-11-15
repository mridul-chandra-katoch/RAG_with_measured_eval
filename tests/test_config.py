import unittest
from pathlib import Path
from ragkit.config import CorpusConfig

class TestConfig(unittest.TestCase):
    def test_load_yaml(self):
        cfg_path = Path("configs/corpus.yaml")
        if not cfg_path.exists():
            self.skipTest("corpus.yaml not created yet")
            
        cfg = CorpusConfig.load(cfg_path)
        self.assertEqual(cfg.corpus_name, "demo")
        self.assertEqual(cfg.max_tokens, 400)
        self.assertEqual(cfg.retrieval_top_k, 20)
        self.assertEqual(cfg.abstain_mode, "learned")

if __name__ == "__main__":
    unittest.main()