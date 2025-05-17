import streamlit as st
import os
import tempfile
import upload_book
import summarize_book
import qa_session
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage


# Streamlit UI
st.set_page_config(page_title="Book Q&A", layout="wide")
st.title("✍️📖💬 Ask the Author")

# Get some info
st.subheader("📤 Upload your book")

# Ask for book details
book_title = st.text_input("Enter the **Book Title**:")
book_author = st.text_input("Enter the **Author's Name**:")

title_clean, author_clean = upload_book.clean_name(book_title), upload_book.clean_name(book_author)


######################## Storing the pdf file locally #####################
if "data_upload_status" not in st.session_state:

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
            st.info(f"Done creating embedding for {book_title} by {book_author}")

        else:
            st.info(f"We already have embedding for {book_title} by {book_author} saved!")
            
        info_dict = {
                            "book_clean":title_clean,
                            "author_clean":author_clean,
                            "book":book_title,
                            "author":book_author
                    }
        
        upload_book.save_info(info_dict)
        st.session_state["data_upload_status"] = True
        st.session_state["config"] = config


##################### Summarize the book #####################
if st.session_state.get("data_upload_status"):
    if "summary" not in st.session_state:
        with st.spinner(f"{book_author} is thinking...", show_time=True):
            summary = summarize_book.main()
            st.session_state["summary"] = summary
            st.session_state["summary_status"] = True
    else:
        summary = st.session_state["summary"]

    st.divider()
    st.subheader(f"🧠 Summary of {book_title} by {book_author}")
    st.write(summary)


##################### QA session #####################

if st.session_state.get("summary_status"):
    st.divider()
    st.subheader(f"💬 Q&A session with the {book_author}")

    if "qa_initialized" not in st.session_state:

        ##### Setup
        persistent_directory = f"../chroma_db/{title_clean}by{author_clean}"
        config = st.session_state["config"]
        db = qa_session.connect_db(persistent_directory, config)
        llm_mistral = qa_session.call_model(config)

        st.session_state.db = db
        st.session_state.llm = llm_mistral
        st.session_state.qa_initialized = True
        st.session_state.persistent_directory = persistent_directory
        st.session_state.conversation = []


    chatbot = st.container(border=True)
    if user_question := st.chat_input("Enter your question:"):
        with chatbot:
            for q, a in st.session_state.conversation:
                st.chat_message("user").write(q)
                st.chat_message("assistant").write(f"{book_author}: {a}")

        chatbot.chat_message("user").write(user_question)
        relevant_docs = qa_session.rag_retreiver(
                                                    user_question,
                                                    st.session_state.persistent_directory,
                                                    st.session_state.db,
                                                    st.session_state["config"]
                                                )
        combined_query = qa_session.rag_augment(user_question, relevant_docs)

        messages = [
                        SystemMessage(content="You are a helpful assistant who is expert at translating books into understandable language"),
                        HumanMessage(content=combined_query),
                    ]
        result = st.session_state.llm.invoke(messages)
        answer = result.content
        st.session_state.conversation.append((user_question, answer))

        chatbot.chat_message("assistant").write(f"{book_author}: {answer}")