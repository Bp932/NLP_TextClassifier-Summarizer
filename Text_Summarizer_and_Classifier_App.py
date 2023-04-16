import streamlit as st
from PyPDF2 import PdfReader
from nlp_utils2 import *
import pandas as pd
import docx2txt
import spacy
from spacy import displacy
from bs4 import BeautifulSoup
from urllib.request import urlopen
from textblob import TextBlob
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
HTML_WRAPPER = """<div style="overflow-x: auto; border: lpx solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

st.set_page_config(layout="wide")

lst_stopwords = create_stopwords()

# Clean the input text taken from input box in UI. From Choice Summarize Text
def text_cleaning(dtf):
    dtf=add_preprocessed_text(dtf, column="text", 
                            punkt=True, lower=True, slang=True, lst_stopwords=lst_stopwords, lemm=True)
    return dtf

def summary_text_using_textrank (dtf):
    result=textrank(corpus=dtf["text"],ratio=30/364)
    return result

def summary_text_using_bart (dtf):
    result=bart_ui(corpus=dtf["text"])
    return result
    
# Extract text from PDF using pyPDF2.
def extract_text_from_pdf(file_path):
    with open(file_path,'rb') as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text

@st.cache_data
def analyze_text(text):
    nlp = spacy.load('en_core_web_sm')
    return nlp(text)

@st.cache_data
def get_text(raw_url):
    page = urlopen(raw_url)
    soup = BeautifulSoup(page)
    #fetched_text = ''.join(map(lambda p:p.text,soup.find_all(p)))
    fetched_text=' '.join([p.text for p in soup.find_all('p')])
    return fetched_text

def convert_to_df(sentiment):
    sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
    return sentiment_df

def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)
        
        
            
    result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
    return res
               

choice = st.sidebar.selectbox("Select your choice",["Summarize Text","Summarize Document","NER Checker","NER for URL","Sentiment Analysis"])
st.header("App developed for showing NLP Project for BIG DATA LAB")
if choice == "Summarize Text":
    st.subheader("Summarize Text using text summarizer api")
    input_text = st.text_area("Enter your text here.")
    input_text_list = [input_text]
    dtf = pd.DataFrame(input_text_list, columns=['text'])
    if input_text is not None:
        if st.button("Summarize Text"):
            col1,col2 = st.columns([1,1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                input_text_dtf = text_cleaning(dtf)
                result_textrank = summary_text_using_textrank(input_text_dtf)
                st.markdown("**Summarized Text (Extractive Technique - Using TextRank/TF-IDF)**")
                st.success(''.join(result_textrank))
                result_bart = summary_text_using_bart(input_text_dtf)
                st.markdown("**Summarized Text (Abstractive Technique - Using BART)**")
                st.success(result_bart)
    
elif choice == "Summarize Document":
    st.subheader("Summarize Document using text summarizer api")
    input_file = st.file_uploader("Upload Your Document.",type=["pdf"])
    if input_file is not None:
        if st.button("Summarize Document"):
            with open("doc_file.pdf","wb") as f:
                f.write(input_file.getbuffer())
            col1,col2 = st.columns([1,1])
            extracted_text = extract_text_from_pdf("doc_file.pdf")
            extracted_text_list = [extracted_text]
            dtf = pd.DataFrame(extracted_text_list, columns=['text'])
            with col1:
                st.markdown("**Extracted Text from Document**")
                st.info(extracted_text)
            with col2:
                input_text_dtf = text_cleaning(dtf)
                result_textrank = summary_text_using_textrank(input_text_dtf)
                st.markdown("**Summarized Text (Extractive Technique - Using TextRank/TF-IDF)**")
                st.success(''.join(result_textrank))
                result_bart = summary_text_using_bart(input_text_dtf)
                st.markdown("**Summarized Text (Abstractive Technique - Using BART)**")
                st.success(result_bart)
                
elif choice == "NER Checker":
    st.subheader("Entity Recognition")
    raw_text = st.text_area("Enter Text Here","Enter your text here.")
    if st.button("Analyze"):
        docx = analyze_text(raw_text)
        html = displacy.render(docx,style='ent')
        html = html.replace("\n\n","\n")
        #st.write(html,unsafe_allow_html=True)
        st.markdown(html,unsafe_allow_html=True)

elif choice == "NER for URL":
    st.subheader("Analyze text from URL")
    raw_url = st.text_input("Enter URL","Type here")
    text_length = st.slider("Length to Preview",50,100)
    if st.button("Extract"):
        if raw_url != "Type here":
            result = get_text(raw_url)
            len_of_full_text = len(result)
            len_of_short_text = round(len(result)/text_length)

            st.write(result[:len_of_short_text])
            extracted_text_list = [result]
            dtf = pd.DataFrame(extracted_text_list, columns=['text'])
            extracted_text_dtf = text_cleaning(dtf)
            summary_docx = summary_text_using_textrank(extracted_text_dtf)
            summary_docx=''.join(summary_docx)
            docx = analyze_text(summary_docx)
            html = displacy.render(docx,style='ent')
            html = html.replace("\n\n","\n")
            #st.write(html,unsafe_allow_html=True)
            st.markdown(html,unsafe_allow_html=True)

elif choice == "Sentiment Analysis":
    st.subheader("Analyze the sentiments & objectivity of the Text")
    raw_text = st.text_area("Enter Text Here","Enter your text here.")
    if st.button("Analyze"):
        col1,col2 = st.columns(2)
        with col1:
            st.info("Results")
            sentiment = TextBlob(raw_text).sentiment
            st.write(sentiment)
            
            if sentiment.polarity >0 :
                st.markdown("Sentiment:: POSITIVE : 🤗")
            elif sentiment.polarity <0:
                st.markdown("Sentiment:: NEGATIVE : 😠")
            else:
                st.markdown("Sentiment:: NEUTRAL : 😐")
                
            result_df = convert_to_df(sentiment)
            st.dataframe(result_df)
                
            c = alt.Chart(result_df).mark_bar().encode(
                x='metric',
                y='value',
                color='metric')
            st.altair_chart(c,use_container_width=True)
                
                
                
                
        with col2:
            st.info("Token Sentiment")
            token_sentiments = analyze_token_sentiment(raw_text)
            st.write(token_sentiments)