import logging
import re

from pypdf import PdfReader
import pdfplumber

from utils import Utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PDFParser:
    def __init__(self, pdf_file, json_output):
        self.pdf_file = pdf_file
        self.json_output = json_output

    def _extract_text_from_pdf(self):
        reader = PdfReader(self.pdf_file)
        text = ""
        with pdfplumber.open(self.pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + " " if page.extract_text() else ""
                if "243" in page.extract_text():
                    print(page.extract_text())
        logging.info(
            f"Extracted text from {self.pdf_file}, {len(reader.pages)} pages processed."
        )
        return text

    @staticmethod
    def _clean_text(text):
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _split_into_sliding_paragraphs(self, pdf_text, overlap=100, window_size=200):
        pdf_words = pdf_text.split()
        len_pdf = len(pdf_words)
        windows = []
        start_index = 0

        while start_index <= len_pdf:
            end_index = min(start_index + window_size, len_pdf)
            window_text = " ".join(pdf_words[start_index:end_index])
            window_info = {
                "text": window_text,
                "window_num": len(windows) + 1,
            }
            windows.append(window_info)
            start_index += window_size - overlap
        return windows

    def parse_and_save(self):
        pdf_text = self._extract_text_from_pdf()
        cleaned_text = self._clean_text(pdf_text)
        paragraphs = self._split_into_sliding_paragraphs(cleaned_text)
        Utils.write_json(self.json_output, paragraphs)
        logging.info(
            f"Successfully processed {len(paragraphs)} paragraphs and saved to {self.json_output}."
        )


if __name__ == "__main__":
    pdf_parser = PDFParser("data/cod-rutier.pdf", "data/paragraphs_v1.json")
    pdf_parser.parse_and_save()
