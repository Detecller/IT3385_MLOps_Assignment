import streamlit as st


st.title("About this App")
st.write("")
st.markdown("""
<div style='text-align: justify'>
<p>This app hosts two models to predict whether one has <b>Alzheimer's Disease</b> and <b>Lung Cancer</b> separately.</p>
<p>Batch uploading was implemented to support the prediction of multiple patient records at once,
    allowing users to submit a CSV file with many entries, automatically generate predictions and confidence scores for each record,
    and download the results in a single file.
</p>
</div>
""", unsafe_allow_html=True)