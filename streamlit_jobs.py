import sys
import streamlit as st

st.write("Python version:", sys.version)

try:
    import spacy
    st.success("spaCy imported successfully")
except Exception as e:
    st.error("spaCy import failed")
    st.write(e)
