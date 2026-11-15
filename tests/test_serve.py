import unittest
from ragkit.serve import SERVER_UPLOAD_TOKEN, MAX_UPLOAD_SIZE

class TestServe(unittest.TestCase):
    def test_upload_token_length(self):
        # Ensure token is 32 characters (secrets.token_hex(16))
        self.assertEqual(len(SERVER_UPLOAD_TOKEN), 32)

    def test_max_upload_size(self):
        self.assertEqual(MAX_UPLOAD_SIZE, 5 * 1024 * 1024)

if __name__ == "__main__":
    unittest.main()