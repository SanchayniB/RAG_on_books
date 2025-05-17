
import os
import yaml
import argparse
from dotenv import load_dotenv
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_mistralai.chat_models import ChatMistralAI


def read_config(name=''):
    if name == '':
        name = "config.yaml"
    with open(f"{name}", "r") as f:
        config = yaml.load(f,Loader=yaml.Loader)
    
    return config

def load_embedding_model(config):
    embeddings = OllamaEmbeddings(model=config['llm_embedding_model'])
    return embeddings

def connect_db(persistent_directory, config):
    embeddings = load_embedding_model(config)
    vectordb = Chroma(
                            persist_directory=persistent_directory,
                            embedding_function=embeddings
                    )
    return vectordb

def rag_retreiver(query, persistent_directory, vectordb, config):

    #connecting to db
    vectordb = connect_db(persistent_directory,config)
    # Retrieve relevant documents based on the query
    retriever = vectordb.as_retriever(
                                            search_type=    "similarity_score_threshold",
                                            search_kwargs=  {   "k": config['retriever']['k'], 
                                                                "score_threshold": config['retriever']['score_threshold']},
                                        )
    relevant_docs = retriever.invoke(query)

    return relevant_docs

def rag_augment(query, relevant_docs):


    # Combine the query and the relevant document contents
    combined_query = (
                            "Here are some documents that might help answer the question: "
                            + query
                            + "\n\nRelevant Documents:\n"
                            + "\n\n".join([doc.page_content for doc in relevant_docs])
                            + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'. \
                                Please provide the answer in simple language understood by everyday reader"
                    )
    
    return combined_query


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

def main():
    curr_dir = os.getcwd()
    os.chdir(curr_dir)

    config = read_config()
    input_config = read_config(name='input_config.yaml')
    book_clean, author_clean = input_config['book_clean'], input_config['author_clean'] 
    persistent_directory = f"../chroma_db/{book_clean}by{author_clean}"

    db = connect_db(persistent_directory, config)
    # call llm chat model
    llm_mistral=call_model(config)
    
    while True:
        #print("\n\n-------------------------------")
        question = input("Ask your question (q to quit): ")
        if question == "q":
            break

        relevant_docs = rag_retreiver(question, persistent_directory, db, config)
        combined_query = rag_augment(question, relevant_docs)

        messages = [
                        SystemMessage(content="You are a helpful assistant who is expert at translating history books into understandable language."),
                        HumanMessage(content=combined_query),
                    ]
        
        # Invoke the model with the combined input
        result = llm_mistral.invoke(messages)
        #print(f'Answer: {result.content}')
        #return result.content

if __name__ == "__main__":
    main()