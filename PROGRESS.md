# Project Progress

## Phase A: Fix What Exists
- [x] **A1**: Abstainer fallback for small corpora.
  - Status: Done
  - Test: `tests/test_generate.py` passes.
  - Open problems: None.
  - [x] **A2**: Rewrite router in agent.py.
  - Status: Done
  - Test: `tests/test_agent.py` passes (4 tests).
  - Open problems: None.
- [x] **A4**: Rewrite red team suite.
  - Status: Done
  - Test: `python -m ragkit redteam` (3/3 passed). `--no-defense` (2/3 passed).
  - Open problems: None.
- [x] **A5**: Harden server (size cap, rate limit, 0.0.0.0 for Docker).
  - Status: Done
  - Test: Server starts and prints a secure 32-char token.
  - Open problems: None.