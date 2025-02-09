from pathlib import Path
from glob import glob
from typing import List
from spire import doc
from accord import logger
from spire.doc.common import *
from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader


TEXT_FILE_EXTENSION = ".txt"
PDF_EXTENSION = ".pdf"
MD_FILE_EXTENSION = ".md"
DOCX_EXTENSION = ".docx"
DOC_EXTENSION = ".doc"


def extract_pdf_content(file_path: Path) -> List[Document]:
    """
    Extract text content from pdf file
    Args:
        file_path (Path): Path to the pdf file
    Returns:
    List[Document]: Text content of the pdf file
    """
    return PyPDFLoader(file_path).load_and_split()

def extract_docx_content(file_path: Path) -> List[Document]:
    """
    Extract text content from docx file
    Args:
        file_path (Path): Path to the docx file
    Returns:
    List[Document]: Text content of the docx file
    """
    return Docx2txtLoader(file_path).load_and_split()


def extract_doc_content(file_path: Path) -> List[Document]:
    """
    Extract text content from doc file
    first convert the doc file to pdf file then extract text content
    Args:
        file_path (Path): Path to the doc file
    Returns:
    List[Document]: Text content of the doc file
    """
    
    pdf_file_path = file_path.with_suffix(".pdf")
    if not pdf_file_path.exists():
        document = doc.Document()
        document.LoadFromFile(str(file_path))
        document.SaveToFile(str(pdf_file_path), doc.FileFormat.PDF)

    return PyPDFLoader(pdf_file_path).load_and_split()


def extract_text_content(file_path: Path) -> List[Document]:
    """
    Extract text content from text file
    Args:
        file_path (Path): Path to the text file
    Returns:
    List[Document]: Text content of the text file
    """
    return TextLoader(file_path).load_and_split()



def load_file(file_path: Path) -> List[Document]:
    """
    Load file content based on file extension
    Args:
        file_path (Path): Path to the file
    Raises:
        ValueError: If file extension is not allowed
    Returns:
    List[Document]: Text content of the file
    """

    file_extension = Path(file_path.name).suffix

    if file_extension == TEXT_FILE_EXTENSION or file_extension == MD_FILE_EXTENSION:
        return extract_text_content(file_path)
    elif file_extension == PDF_EXTENSION:
        return extract_pdf_content(file_path)
    elif file_extension == DOCX_EXTENSION:
        return extract_docx_content(file_path)
    elif file_extension == DOC_EXTENSION:
        return extract_doc_content(file_path)

    return []


def load_directory_files(directory_path: Path) -> List[Document]:
    """
    Load content of all files in a directory
    Args:
        directory_path (Path): Path to the directory
    Returns:
    List[Document]: Text content of all files in the directory
    """

    docs = []
    file_paths = glob(f"{directory_path}/*")
    if os.path.join(os.path.split(file_paths[0])[0],'info.md') not in file_paths:
        logger.error(f"info.md file not found in {directory_path}")
        return docs
    else:
        file_paths.remove(os.path.join(os.path.split(file_paths[0])[0],'info.md'))
        
    file_paths = sorted(file_paths)
    for file_path in file_paths:
        docs.extend(load_file(Path(file_path)))
    return docs