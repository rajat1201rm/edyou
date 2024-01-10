from txtai.embeddings import Embeddings
from openai import OpenAI
import nltk
from PyPDF2 import PdfReader
import sys
from nltk.tokenize import word_tokenize
import os
import shutil
import pickle
from dotenv.main import load_dotenv, dotenv_values
from flask import Flask, render_template, request,redirect, url_for,Blueprint,session
from flask import jsonify,render_template,Blueprint
from flask_cors import CORS
from datetime import datetime
import os
from docx import Document

import streamlit as st
#from translate import translate_corpus
load_dotenv()
openai_api=os.getenv("OPENAI_API")
doc_source = os.getenv("DOC_DIRECTORY")


def load_index():
    embeddings = Embeddings(hybrid = True,content=True,keyword = True)
    embeddings.load("index")
    return embeddings


def empty_directory(directory_path):
    
        # Remove all files and subdirectories within the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)

        
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api,
)       


def user_query(e,question_asked,brain_name):
        string_tosearch=" "
        res =e.search(question_asked,10)
        for i in range(len(res)):
            print("\nEmbedding Line ------------>"+res[i]['id'])
            print("Content -----------> \n"+res[i]['text'])
            print("Score----------------> ", res[i]['score'])
        
            string_tosearch+=(res[i]['text'])
        print("\n\nAnswer <------------------------------------------------>\n\n")
        print("GPT REPLY --------------------------> ")
        return gpt_reply(question_asked+"strictly answer in very precise words -"+string_tosearch,brain_name)
        
        
def gpt_reply(user_input,brain_name):
    res_box = st.empty()
    report =[]
    for resp in client.chat.completions.create(
      model="gpt-4",
      messages=[{"role":"system",'content':f'0-> talk in first person only.Never Reply Like I {brain_name} or As {brain_name} for answers. 1-> Only Answer for what is Asked  1 ->.2-> You are {brain_name}, . 3.-> You are Not AI Model You are {brain_name} 4.-> Look for content in Prompt to paraphrase and in case you cannot answer , say "Hey ,I might not be able to help you with that".' },
          {"role": "user", "content": user_input}],
      stream=True,
      temperature =0.2,
      
    ):
        report.append(resp.choices[0].delta.content or "")
        result = "".join(report).strip()
        result =result.replace("\n","")
        res_box.markdown(f'*{result}*')
    
#   string =" "
#   for x in stream :
#     string += (x.choices[0].delta.content or "")
#   return string
    

# def gpt_reply(user_input):
  
#   for response in client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "system", "content": "Answer in first person of the language the question is asked in."},
#                   {"role": "user", "content": user_input}

                  
#                   ],
#         stream=True,
#         temperature =0,
#     ):
#       sys.stdout.write(response.choices[0].delta.content or "")
#       sys.stdout.flush()



#############################################-------------------------train file code  ---------------#############################################################

def get_pdf_text(pdf_docs):
    # print("Reading File  - " +  pdf_docs)
    text=""
    pdf_in = open(pdf_docs, 'rb')
    reader = PdfReader(pdf_in)
    for i in range(0,len(reader.pages)):

        page = reader.pages[i]
        text+= page.extract_text()
    text = ' '.join(text.split('\n'))
    return text

def get_txtfile_text(text_doc_path):
    with open(text_doc_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def get_docxfile_text(docx_file):
    doc = Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def load_all_text():
    filedata = ""
    doc_dir=os.getenv("DOC_DIRECTORY")
    doc_list = os.listdir(doc_dir)
    print("Files - ",doc_list)
    for file_name in doc_list:
        if file_name.endswith('.pdf'):
            filedata=filedata+get_pdf_text(doc_dir+"/"+file_name)
        elif file_name.endswith('.txt'):
            filedata=filedata+get_txtfile_text(doc_dir+"/"+file_name)
        elif file_name.endswith('.docx'):
            filedata=filedata+get_docxfile_text(doc_dir+'/'+file_name)

    # print(filedata)
    return filedata

#print(load_doc_text())

def split_text(input_text, words_per_chunk):
    words = word_tokenize(input_text)
    chunks = [words[i:i + words_per_chunk] for i in range(0, len(words), words_per_chunk)]
    return chunks

def create_embeddings():

    print("Loading Text Corpus")
    content  = load_all_text()
    word_chunks = split_text(content, 100)
    for i in range(len( word_chunks)):
        string=""
        for x in word_chunks[i]:
            string = string+x+" "
        word_chunks[i] =string
    #print(word_chunks)
    txtai_data=[]
    i=0
    for x in word_chunks:
        txtai_data.append((i,x,None))
        i+=1
    #print(txtai_data[10])
    print("Creating indexes")
    with open('./arraydata.pkl', 'wb') as file:
        pickle.dump(txtai_data, file)
    
    embeddings = Embeddings(hybrid = True,content=True,keyword = True)
    with open('arraydata.pkl', 'rb') as file:
        data = pickle.load(file)
        print("Data loaded from Array File")
    
    print("Training called")
    embeddings.index(data)
    embeddings.save("index")
    empty_directory(doc_source)
    

def upload_files_to_folder(files, folder_path):
    
    for file in files:
        file_path = os.path.join(folder_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        print(f"File saved: {file.name}")
    
def main():
    st.set_page_config(page_title="Chat With Your Personal Bot")
    

    #with st.sidebar:
    st.image('Images/vdoit_technologies_private_limited_logo.jpeg')
    st.header("TxtAI")
    st.text("Version Alpha")
    brain_name = st.text_input("Mention The Personality Name")
    st.subheader("Ingest Your documents")
    files = st.file_uploader(
        "Upload your files here and click on 'Process'", accept_multiple_files=True)
    
    if st.button("Upload File"):
        if files:
            upload_files_to_folder(files, doc_source)


    if st.button("Train Your System"):
        with st.spinner("Processing"):
            create_embeddings()
    e =load_index()

    user_question = st.text_input("Ask TxtAi A Question:")
    
    if st.button("Query Your Documents"):
        if "you" in user_question.lower():
            user_question=user_question.replace("you",brain_name)
        print(user_question)
        user_query(e,user_question,brain_name)
        


if __name__ == "__main__":
    main()