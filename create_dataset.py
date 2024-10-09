import logging
import re

from pypdf import PdfReader

from utils import write_json

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
        for page in reader.pages:
            text += page.extract_text() + " " + f"[PAGE_NUM={page.page_number}]"
        logging.info(
            f"Extracted text from {self.pdf_file}, {len(reader.pages)} pages processed."
        )
        return text

    @staticmethod
    def _extract_page_numbers(window_text):
        pattern = r"\[PAGE_NUM=(\d+)\]"
        page_numbers = re.findall(pattern, window_text)
        updated_window_text = re.sub(pattern, "", window_text)
        unique_int_pages = list(set([int(num) for num in page_numbers]))
        return unique_int_pages, updated_window_text.strip()

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
            page_nums, updated_window_text = self._extract_page_numbers(window_text)
            window_info = {
                "text": updated_window_text,
                "page_num": page_nums,
                "window_num": len(windows) + 1,
            }
            windows.append(window_info)
            start_index += window_size - overlap
        return windows

    def parse_and_save(self):
        pdf_text = self._extract_text_from_pdf()
        cleaned_text = self._clean_text(pdf_text)
        paragraphs = self._split_into_sliding_paragraphs(cleaned_text)
        write_json(self.json_output, paragraphs)
        logging.info(
            f"Successfully processed {len(paragraphs)} paragraphs and saved to {self.json_output}."
        )


if __name__ == "__main__":
    pdf_parser = PDFParser("data/cod-rutier.pdf", "data/paragraphs.json")
    pdf_parser.parse_and_save()
