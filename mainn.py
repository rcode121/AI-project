from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import openai 
import PyPDF2
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import dotenv

import os 
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'


# Python standard libraries
import json
import os
import sqlite3

# Third party libraries
from flask import Flask, redirect, request, url_for
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from oauthlib.oauth2 import WebApplicationClient
import requests

# Internal imports
from db import init_db_command
from user import User

# Configuration
GOOGLE_CLIENT_ID = "569662328285-i4a3tq7v7ipapo5dfj8n0vdlfjaesmld.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"
GOOGLE_DISCOVERY_URL = (
    "https://accounts.google.com/.well-known/openid-configuration"
)


config = dotenv.dotenv_values(".env")
openai.api_key = config['OPENAI_API_KEY']



app = Flask(__name__)
#app.secret_key = 'your secret key'  # Replace with your secret key
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)


# User session management setup
# https://flask-login.readthedocs.io/en/latest
login_manager = LoginManager()
login_manager.init_app(app)


@login_manager.unauthorized_handler
def unauthorized():
    return "You must be logged in to access this content.", 403


# Naive database setup
try:
    init_db_command()
except sqlite3.OperationalError:
    # Assume it's already been created
    pass
# OAuth2 client setup
client = WebApplicationClient(GOOGLE_CLIENT_ID)


# Flask-Login helper to retrieve a user from our db
@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


conversation_chains = {}

def get_pdf_text(filename):
    text = ""
    with open(filename, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text()        
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(chunks):
    openai.api_key = config['OPENAI_API_KEY'] 
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain





@app.route('/api/chat_without_file', methods=['POST'])
def chat_without_file():
    if 'message' not in request.json:
        return jsonify(error='No message in the request'), 400
    message = request.json['message']  
    print(message)                                  # this is assumed to be included in the POST data
    chat_models = "gpt-3.5-turbo"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=chat_models,
        messages=messages
    )
    return jsonify(message=response['choices'][0]['message']['content'])

@app.route('/api/upload_without_message', methods=['POST'])
def upload_without_message():
    if 'file' not in request.files:
        return jsonify(error='No file part in the request'), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No file selected for uploading'), 400
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file.save(filename)

        text = get_pdf_text(filename)
        chunks = get_text_chunks(text)
        vectorstore = get_vectorstore(chunks)
        conversation_chains = get_conversation_chain(vectorstore)
        query = "Write a summary of the  uploaded file"
        response = conversation_chains({"question": query}) # you'll have to adapt this line to how the method is actually called
        print(response)
        os.remove(filename)
        return jsonify(message = response['answer'])
    else:
        return jsonify(error='Allowed file type is pdf'), 400
    

@app.route('/api/chat_with_file', methods=['POST'])
def chat_with_file():
    if 'file' not in request.files or 'message' not in request.form:
        return jsonify(error='No file or message part in the request'), 400
    file = request.files['file']
    message = request.form['message']
    #filename = request.json['filename']  # this is assumed to be included in the POST data

    filename = secure_filename(file.filename)
    file.save(filename)

    text = get_pdf_text(filename)
    chunks = get_text_chunks(text)
    vectorstore = get_vectorstore(chunks)
    conversation_chains = get_conversation_chain(vectorstore)
    query = message
    print(message)
    response = conversation_chains({"question": query}) # you'll have to adapt this line to how the method is actually called
    print(response)
    os.remove(filename)
    return jsonify(message = response['answer'])


@app.route("/")
def index():
    if current_user.is_authenticated:
        return render_template('index.html')
    else:
        return render_template('login.html')
    
    

@app.route("/style.css")
def styles():
    return send_from_directory("static", "style.css")    


@app.route("/login")
def login():
    # Find out what URL to hit for Google login
    google_provider_cfg = get_google_provider_cfg()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]

    # Use library to construct the request for login and provide
    # scopes that let you retrieve user's profile from Google
    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=request.base_url + "/callback",
        scope=["openid", "email", "profile"],
    )
    return redirect(request_uri)


@app.route("/login/callback")
def callback():
    # Get authorization code Google sent back to you
    code = request.args.get("code")

    # Find out what URL to hit to get tokens that allow you to ask for
    # things on behalf of a user
    google_provider_cfg = get_google_provider_cfg()
    token_endpoint = google_provider_cfg["token_endpoint"]

    # Prepare and send request to get tokens! Yay tokens!
    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.base_url,
        code=code,
    )
    token_response = requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
    )

    # Parse the tokens!
    client.parse_request_body_response(json.dumps(token_response.json()))

    # Now that we have tokens (yay) let's find and hit URL
    # from Google that gives you user's profile information,
    # including their Google Profile Image and Email



    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]


    userinfo_endpoint = "https://www.googleapis.com/oauth2/v3/userinfo"

    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)
    for i in userinfo_response:
        print(i)
    # We want to make sure their email is verified.
    # The user authenticated with Google, authorized our
    # app, and now we've verified their email through Google!
    if userinfo_response.json().get("email_verified"):
        unique_id = userinfo_response.json()["sub"]
        users_email = userinfo_response.json()["email"]
        picture = userinfo_response.json()["picture"]
        users_name = userinfo_response.json()["given_name"]
    else:
        return "User email not available or not verified by Google.", 400

    # Create a user in our db with the information provided
    # by Google
    user = User(
        id_=unique_id, name=users_name, email=users_email, profile_pic=picture
    )

    # Doesn't exist? Add to database
    if not User.get(unique_id):
        User.create(unique_id, users_name, users_email, picture)

    # Begin user session by logging the user in
    login_user(user)

    # Send user back to homepage
    return redirect(url_for("index"))


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

def get_google_provider_cfg():
    return requests.get(GOOGLE_DISCOVERY_URL).json()

if __name__ == '__main__':
    app.run(debug=True, port=5000)






