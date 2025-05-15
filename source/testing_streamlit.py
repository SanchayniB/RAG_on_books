import streamlit as st
import os
import tempfile
import upload_book
from langchain_chroma import Chroma


# Streamlit UI
st.set_page_config(page_title="Book Q&A", layout="wide")
st.title("📚 Ask Questions About Any Book")


# Ask for book details
book_title = st.text_input("Enter the **Book Title**:")
book_author = st.text_input("Enter the **Author's Name**:")

title_clean, author_clean = upload_book.clean_name(book_title), upload_book.clean_name(book_author)

########## Storing the pdf file locally
uploaded_file = st.file_uploader("Upload a novel (PDF)", type=["pdf"])
if uploaded_file:
    with st.spinner("Reading the PDF..."):
        config = upload_book.read_config()
        
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        docs = upload_book.read_pdf(tmp_path,config)
        st.success("PDF loaded successfully!")

    
    persistent_directory = f"../chroma_db/{title_clean}by{author_clean}"
    document_exists = os.path.exists(persistent_directory)
    if not document_exists:
        embeddings = upload_book.load_embedding_model(config=config)

        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        st.text(f"\n---Done creating embedding for {book_title} by {book_author}---")

    else:
        st.text(f"\n---We already have embedding for {book_title} by {book_author} saved!---")
        
        info_dict = {
                            "book_clean":title_clean,
                            "author_clean":author_clean,
                            "book":book_title,
                            "author":book_author
                    }
        
        upload_book.save_info(info_dict)

    