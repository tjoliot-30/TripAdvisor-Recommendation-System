with open("pdf_content.txt", "r", encoding="utf-16le") as f:
    text = f.read()
with open("pdf_content_utf8.txt", "w", encoding="utf-8") as f:
    f.write(text)
