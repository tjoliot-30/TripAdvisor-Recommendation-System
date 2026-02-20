import sys
try:
    import pypdf
except ImportError:
    print("pypdf not installed.")
    sys.exit(1)

reader = pypdf.PdfReader("Information Retrieval Instruction Project1 2026.pdf")
for i, page in enumerate(reader.pages):
    print(f"--- PAGE {i+1} ---")
    print(page.extract_text())
