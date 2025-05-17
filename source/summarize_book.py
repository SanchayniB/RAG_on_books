import os
import yaml
import argparse
import re
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from pathlib import Path



def read_config(name=''):
    if name == '':
        name = "config.yaml"
    with open(f"{name}", "r") as f:
        config = yaml.load(f,Loader=yaml.Loader)
    
    return config


def load_embedding_model(config):
    embeddings = OllamaEmbeddings(model=config['llm_embedding_model'])
    return embeddings

def call_model(config):
    dotenv_path = Path(config['env_path'])
    load_dotenv(dotenv_path=dotenv_path)

    MY_KEY = os.getenv('MISTRAL_KEY')
    os.environ["MISTRAL_API_KEY"] = MY_KEY

    llm_mistral = ChatMistralAI(
                                    model=config['llm_chat_model'],
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
   
    # print(result['output_text'])
    return result['output_text']

def main():
    curr_dir = os.getcwd()
    os.chdir(curr_dir)

    config = read_config()
    input_config = read_config(name='input_config.yaml')
    book_clean, author_clean = input_config['book_clean'], input_config['author_clean'] 
    book, author = input_config['book'], input_config['author'] 

    persistent_directory = f"../chroma_db/{book_clean}by{author_clean}"

    #print(f"\n-------------{book} by {author}-----------")
    result = get_summary(persistent_directory=persistent_directory,config=config)
    return result



if __name__ == "__main__":

    main()