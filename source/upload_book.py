import os
import yaml
import argparse
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def read_pdf(path, config):
    # Read the text content from the file
    loader = PyPDFLoader(path)
    documents = loader.load_and_split()
    # Split the document into chunks
    text_splitter = CharacterTextSplitter(  chunk_size=config['document_chunking']['chunk_size'],
                                            chunk_overlap=config['document_chunking']['chunk_overlap']
                                        )
    docs = text_splitter.split_documents(documents)

        # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")


    return docs

def load_embedding_model(config):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings

def read_config(name=''):
    if name == '':
        name = "config.yaml"
    with open(f"{name}", "r") as f:
        config = yaml.load(f,Loader=yaml.Loader)
    
    return config

def save_info(info_dict, filename=''):
    
    if filename=='':
        filename="input_config"
    
        
    with open(f"{filename}.yaml") as f:
        config = yaml.safe_load(f)

    for key,val in info_dict.items():
        config[f'{key}'] = val

    with open(f"{filename}.yaml", "w") as f:
        yaml.dump(config, f)
    
    print('\n---config file updated---')

def clean_name(name):
    name = name.replace(" ", "")
    name = re.sub("[^a-zA-Z]+", "", name)
    name = name.lower()

    return name

def main(book, author):
    curr_dir = os.getcwd()
    os.chdir(curr_dir)

    config = read_config()
    book_clean, author_clean = clean_name(book), clean_name(author)
    print(book_clean, author_clean)

    path = f"../data/{book_clean}by{author_clean}.pdf"
    persistent_directory = f"../chroma_db/{book_clean}by{author_clean}"

    document_exists = os.path.exists(persistent_directory)
    if not document_exists:
        docs = read_pdf(path=path,config=config)
        embeddings = load_embedding_model(config=config)

        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        print(f"\n---Done creating embedding for {book} by {author}---")

    else:
        print(f"\n---We already have embedding for {book} by {author} saved!---")

    info_dict = {
                        "book_clean":book_clean,
                        "author_clean":author_clean,
                        "book":book,
                        "author":author
                 }
    
    save_info(info_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--book", dest='book', type=str, help="Name of the book")
    parser.add_argument("--author", dest='author', type=str, help="Author of the book")
    args = parser.parse_args()

    print("\n--- Selected Book ---")
    print("Book:", args.book)
    print("By Author:", args.author)

    main(book=args.book, author=args.author)


