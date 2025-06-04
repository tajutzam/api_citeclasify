import fitz
import re

def extract_keywords_from_pdf(pdf_path):
    """Ekstrak keyword dari file PDF, termasuk jika ada di baris setelah 'Kata Kunci'"""
    doc = fitz.open(pdf_path)
    keywords = []

    for page in doc:
        lines = page.get_text().split('\n')
        for i, line in enumerate(lines):
            if re.search(r'(?i)^kata kunci$|^keywords$', line.strip()):
                if i + 1 < len(lines):
                    raw_keywords = lines[i + 1]

                    keywords = re.split(r',|;', raw_keywords)
                    keywords = [kw.strip() for kw in keywords if kw.strip()]
                    return keywords

            match = re.search(r'(?i)(kata kunci|keywords)\s*[:]\s*(.+)', line)
            if match:
                raw_keywords = match.group(2)
                keywords = re.split(r',|;', raw_keywords)
                keywords = [kw.strip() for kw in keywords if kw.strip()]
                return keywords

    return keywords




print(extract_keywords_from_pdf('test.pdf'))