from pathlib import Path
from pdfminer.high_level import extract_text

SUPPORTED = {".md", ".txt", ".pdf"}

def load_documents(folder):
    folder = Path(folder)
    for path in folder.rglob("*"):
        if path.suffix.lower() not in SUPPORTED:
            continue
            
        if path.suffix.lower() == ".pdf":
            try:
                # Extract all text from the PDF at once
                text = extract_text(str(path))
                if text and text.strip():
                    for i in range(0, max(1, len(text)), 2000):
                        yield {
                            "file": str(path.relative_to(folder)),
                            "page": i // 2000 + 1,
                            "text": text[i:i + 2000],
                        }
            except Exception as e:
                print(f"Error reading PDF {path.name}: {e}")
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for i in range(0, max(1, len(text)), 2000):
                yield {
                    "file": str(path.relative_to(folder)),
                    "page": i // 2000 + 1,
                    "text": text[i:i + 2000],
                }