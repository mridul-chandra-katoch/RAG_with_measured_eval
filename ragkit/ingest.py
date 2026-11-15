import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from html.parser import HTMLParser
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

SUPPORTED = {".md", ".txt", ".pdf", ".html", ".htm", ".docx"}

class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
    def handle_data(self, data):
        if data.strip():
            self.text.append(data.strip())

def _read_docx(path):
    """Reads text from a .docx file using standard library."""
    try:
        with zipfile.ZipFile(path) as docx:
            xml_content = docx.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            # The namespace for WordprocessingML
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            texts = []
            for p in tree.findall('.//w:p', ns):
                texts.append("".join(node.text for node in p.findall('.//w:t', ns) if node.text))
            return "\n".join(texts)
    except Exception as e:
        print(f"Error reading DOCX {path.name}: {e}")
        return ""

def _read_html(path):
    """Reads text from an HTML file."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        parser = _HTMLTextExtractor()
        parser.feed(text)
        return " ".join(parser.text)
    except Exception as e:
        print(f"Error reading HTML {path.name}: {e}")
        return ""

def _read_pdf(path):
    """Yields text page by page from a PDF."""
    try:
        for page_layout in extract_pages(str(path)):
            text = ""
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text += element.get_text()
            yield text
    except Exception as e:
        print(f"Error reading PDF {path.name}: {e}")

def load_documents(folder):
    """Yields normalized documents with real page numbers."""
    folder = Path(folder)
    for path in folder.rglob("*"):
        if path.suffix.lower() not in SUPPORTED:
            continue
            
        suffix = path.suffix.lower()
        relative_path = str(path.relative_to(folder))
        
        if suffix == ".pdf":
            for page_num, text in enumerate(_read_pdf(path), start=1):
                if text.strip():
                    yield {
                        "file": relative_path,
                        "page": page_num,
                        "text": text.strip(),
                    }
        elif suffix in (".html", ".htm"):
            text = _read_html(path)
            if text.strip():
                yield {"file": relative_path, "page": 1, "text": text.strip()}
        elif suffix == ".docx":
            text = _read_docx(path)
            if text.strip():
                yield {"file": relative_path, "page": 1, "text": text.strip()}
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                yield {"file": relative_path, "page": 1, "text": text.strip()}