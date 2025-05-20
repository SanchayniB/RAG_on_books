import streamlit as st

######################## Setting the background ########################
def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
                    f"""
                    <style>
                    .stApp {{
                        background: url("https://images.unsplash.com/photo-1629968417850-3505f5180761?q=80&w=3904&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
                        background-size: cover
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
    
set_bg_hack_url()