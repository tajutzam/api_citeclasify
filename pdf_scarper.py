import fitz  # PyMuPDF

def extract_title_from_pdf(pdf_file):
    # Membuka file PDF
    doc = fitz.open(pdf_file)
    title = None
    
    # Looping melalui setiap halaman untuk mencari teks dengan format tebal (bold)
    for page in doc:
        blocks = page.get_text("dict")["blocks"]  # Mendapatkan teks dalam format dictionary
        
        for block in blocks:
            if block['type'] == 0:  # Teks biasa
                for line in block['lines']:
                    for span in line['spans']:
                        # Periksa apakah gaya font adalah bold
                        if 'bold' in span['font'].lower():
                            # Ambil teks dari span yang tebal
                            title = span['text']
                            break
            if title:
                break
        if title:
            break

    return title if title else "Title not found in the PDF"

pdf_file_path = "test.pdf"
title = extract_title_from_pdf(pdf_file_path)
print(f"Title found: {title}")
