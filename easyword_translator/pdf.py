import fitz  # PyMuPDF


def read_pdf(file_path):
    # Open the PDF file
    document = fitz.open(file_path)
    text = ""

    # Iterate through the pages
    for page_num in range(len(document)):
        # Extract text from each page
        page = document.load_page(page_num)
        text += page.get_text()

    # Close the PDF document
    document.close()

    return text


def remove_line_breaks(text):
    #  remove only single line breaks, not paragraphs
    # find line breaks and it is not followed by a period
    for i in range(len(text)):
        if i == 0 or i == len(text) - 1:
            continue
        if text[i] == "\n" and text[i - 1] != "." and text[i + 1] != "\n":
            text = text[:i] + " " + text[i + 1 :]
    return text


if __name__ == "__main__":
    file_path = "example3.pdf"
    pdf_text = read_pdf(file_path)
    print(remove_line_breaks(pdf_text))
