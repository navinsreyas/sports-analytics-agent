"""
scripts/build_knowledge.py
---------------------------
ONE-TIME SETUP SCRIPT - run this after datapush.py.

Reads cricket.pdf and football.pdf from sportsdata/, splits them into
overlapping text chunks, embeds them with a local HuggingFace model, and
saves the resulting ChromaDB vector store to sportsdata/chroma_db/.

The RAG tool in agent.py searches this store at query time.

Run from the project root:
    python scripts/build_knowledge.py
"""

import os
import glob
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_FOLDER  = os.path.join(BASE_DIR, 'sportsdata')
VECTOR_DB_PATH = os.path.join(SOURCE_FOLDER, 'chroma_db')


def build_knowledge_base():
    print(f'Targeting folder: {SOURCE_FOLDER}')

    if not os.path.exists(SOURCE_FOLDER):
        print(f'Error: Folder not found: {SOURCE_FOLDER}')
        return

    pdf_files = glob.glob(os.path.join(SOURCE_FOLDER, '*.pdf'))

    if not pdf_files:
        print('Error: No PDF files found!')
        return

    print(f'Found {len(pdf_files)} rulebooks: {[os.path.basename(f) for f in pdf_files]}')

    all_docs = []
    for pdf_file in pdf_files:
        try:
            print(f'Reading: {os.path.basename(pdf_file)}...')
            loader = PyPDFLoader(pdf_file)
            all_docs.extend(loader.load())
        except Exception as e:
            print(f' Could not load {pdf_file}: {e}')

    if not all_docs:
        print(' No text extracted. Exiting.')
        return

    print(f'Processing {len(all_docs)} pages...')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits        = text_splitter.split_documents(all_docs)
    print(f'Created {len(splits)} searchable chunks.')

    try:
        embedding_function = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH)
            print(' (Overwriting old database)')

        Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            persist_directory=VECTOR_DB_PATH,
        )
        print(f'Success! Brain saved to: {VECTOR_DB_PATH}')

    except Exception as e:
        print(f'Error saving database: {e}')


if __name__ == '__main__':
    build_knowledge_base()
