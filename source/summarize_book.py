import os
import yaml
import argparse
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from pathlib import Path



def read_config():
    with open("config.yaml", "r") as f:
        config = yaml.load(f,Loader=yaml.Loader)
    
    return config


def load_embedding_model(config):
    embeddings = OllamaEmbeddings(model=config['llm_model'])
    return embeddings

def call_model(config):
    dotenv_path = Path(config['env_path'])
    load_dotenv(dotenv_path=dotenv_path)

    MY_KEY = os.getenv('MISTRAL_KEY')
    os.environ["MISTRAL_API_KEY"] = MY_KEY

    llm_mistral = ChatMistralAI(
                                    model="mistral-small-latest",
                                    temperature=0
                                )
    return llm_mistral 

def get_summary(persistent_directory,config):
    embeddings=load_embedding_model(config)
    vector_db = Chroma(
                        persist_directory=persistent_directory,
                        embedding_function=embeddings
                    )
    docs = []
    data = vector_db.get()['documents']
    for i in range(len(data)):
        doc = Document(page_content=data[i])
        docs.append(doc)

    llm_mistral=call_model(config)
    summarize_chain = load_summarize_chain(llm=llm_mistral, chain_type="stuff",verbose=True)
    result = summarize_chain.invoke(docs)
    print(result['output_text'])


def main(book, author):
    curr_dir = os.getcwd()
    os.chdir(curr_dir)

    config = read_config()
    persistent_directory = f"../chroma_db/{book}by{author}"

    get_summary(persistent_directory=persistent_directory,config=config)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--book", dest='book', type=str, help="Name of the book")
    parser.add_argument("--author", dest='author', type=str, help="Author of the book")
    args = parser.parse_args()

    print("\n--- Selected Book ---")
    print("Book:", args.book)
    print("By Author:", args.author)

    main(book=args.book, author=args.author)