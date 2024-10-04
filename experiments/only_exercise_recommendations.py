import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import getpass
import os

from Recommender_file import Classifier, Recommender


aiko_ud = '''
    "name": "Aiko",
    "age": 22,
    "proficiency_level": "intermediate",
    "country": "Japan",
    "interests": ["anime", "movies", "technology", "fashion"],
'''
aiko_history = '''
Incorrect verb tense in conditional sentence (Grammar - Verb tenses)
Subject-verb agreement error (Grammar - Subject-verb agreement)
Used incorrect articles (Grammar - Articles)
Failed to link ideas (Fluency - Sentence linking)
Preposition misuse (Grammar - Prepositions)
Inconsistent stress patterns (Pronunciation - Word stress)
Mispronounced verb forms (Pronunciation - Consonant sounds)
'''
lucas_ud = '''
    "name": "Lucas",
    "age": 28,
    "proficiency_level": "beginner",
    "country": "Brazil",
    "interests": ["sports", "cooking", "movies", "travel"],
'''

lucas_history = '''
Used too many filler words like "um" and "uh" (Fluency - Filler words)
Spoke too quickly, making it hard to follow (Fluency - Speaking speed)
Hesitated frequently during speech (Fluency - Hesitation)
Struggled with intonation during questions (Pronunciation - Intonation)
Failed to pause appropriately between ideas (Fluency - Sentence linking)
Used incorrect collocations that sounded awkward (Vocabulary - Collocations)
Inconsistent stress patterns on keywords (Pronunciation - Word stress)
'''

emma_ud = '''
    "name": "Emma",
    "age": 30,
    "proficiency_level": "advanced",
    "country": "United States",
    "interests": ["music", "books", "technology"],
'''

emma_history = '''
Used repetitive vocabulary in discussion (Vocabulary - Word choice)
Misused idiomatic expressions (Vocabulary - Idiomatic expressions)
Mispronounced specific vocabulary terms (Pronunciation - Consonant sounds)
Limited use of synonyms for variety (Vocabulary - Word choice)
Failed to define unfamiliar words when used (Vocabulary - Academic vocabulary)
Used jargon that confused the audience (Vocabulary - Word choice)
Struggled to connect complex ideas smoothly (Fluency - Sentence linking)
'''

api_key = "AIzaSyCPfIdMffhoR2nxre5pmCFuYmvEI6G7oyY"  # 1 pro
# api_key = "AIzaSyB6qvwIDeJeBcQDYB1O_NmABoK9yGs-pEk" # 1.5pro

genai.configure(api_key=api_key)
        
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = api_key

# Load documents
loader = DirectoryLoader(path="Exercises/", glob="**/*.txt")
pages = loader.load()

# Embedding model and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(
            documents=pages, 
            embedding=embeddings
)

classifier_tool = Classifier()

# Title of the app
st.title("Exercise Recommendation System")

selected_user = None
user_ud = None
user_history = None
if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None
if 'user_ud' not in st.session_state:
    st.session_state.user_ud = None
if 'user_history' not in st.session_state:
    st.session_state.user_history = None


# User selection
users = ["Aiko", "Lucas", "Emma", "New User"]
selected_user = st.selectbox("Select a User", users)

# Store the selected user in session state
if selected_user != st.session_state.selected_user:
    st.session_state.selected_user = selected_user
if st.session_state.selected_user == "Aiko":
    user_ud = aiko_ud
    user_history = aiko_history
if st.session_state.selected_user == "Lucas":
    user_ud = lucas_ud
    user_history = lucas_history
if st.session_state.selected_user == "Emma":
    user_ud = emma_ud
    user_history = emma_history

# New User form
if st.session_state.selected_user == "New User":
    temp = None
    with st.form("user_form"):
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=0)
        country = st.text_input("Country")
        proficiency_level = st.text_input("proficiency level")
        interests = st.text_input("Interests")
        submit_button = st.form_submit_button(label='Submit')
        
        if submit_button:
            st.session_state.user_ud = f'''
            "name": {name},
            "age": {age},
            "proficiency_level": {proficiency_level},
            "country": {country},
            "interests": {interests},
            '''
            st.write(f"User {name}, Age {age}, from {country}, interested in {interests}, added successfully!")

# Input box for sentence
sentence = st.text_input("Enter a sentence for recommendation")

# Display outputs if there is an input sentence
# if submit_button:
if st.button("Generate Exercise"):
    st.write("Processing sentence:", sentence)
    
    # Example placeholders for outputs from your functions
    output1 = classifier_tool.get_classifier_response(sentence)
    output2 = classifier_tool.get_corrected_response(sentence)
    recommender_tool = Recommender(output1,vectorstore)
    output3 = recommender_tool.get_exercise(user_ud,user_history)
    

    st.write(output1)
    st.write(f"Corrected Sentence: {output2}")

    st.text_area("userdemo", value=user_ud, height=150)
    st.text_area("userhist", value=user_history, height=150)

    # i = 0
    # st.write("### Exercise Recommendation:")
    # while i:
    #     output3 = recommender_tool.get_exercise(user_ud,user_history)
    #     st.text_area("Recommendations", value=output3, height=150)


    # # For large outputs, using text area for better readability
    st.write("### Exercise Recommendation:")
    st.text_area("Recommendations", value=output3, height=150)
