import os
import PyPDF2
from typing import List


class DocumentLoader:
    def __init__(self, file_directory: str):
        self.directory = file_directory
        self.data: List[str] = []

    def load_documents(self) -> List[str]:
        self.data = []
        for filename in os.listdir(self.directory):
            file_path = os.path.join(self.directory, filename)
            if filename.endswith('.txt'):
                self.data.append(self.load_txt(file_path))
            elif filename.endswith('.pdf'):
                self.data.append(self.load_pdf(file_path))
        return self.data
    
    def load_txt(self, file_path: str) -> str:
        """Load text from .txt file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def load_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF {file_path}: {e}")