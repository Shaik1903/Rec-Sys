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

api_key = "AIzaSyCPfIdMffhoR2nxre5pmCFuYmvEI6G7oyY"  # 1 pro
# api_key = "AIzaSyB6qvwIDeJeBcQDYB1O_NmABoK9yGs-pEk" # 1.5pro
# model_name = "gemini-1.0-pro"
# model_name = "gemini-1.5-pro"
model_name = "gemini-1.5-flash"

genai.configure(api_key=api_key)
        
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = api_key

class Classifier:
    def __init__(self, api_key : str = api_key, model_name: str = model_name):
        
        # Configure API key
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = self.api_key
        
        self.llm = ChatGoogleGenerativeAI(model=model_name,temperature=0.0, google_api_key=self.api_key, max_tokens=None)
        
        # Templates for classification, correction, and exercise recommendation
        self.classifier_template = '''
Imagine you are an expert in English, and your job is to classify the errors in the given sentence into specific categories and sub-categories based on the following:

Categories:
- Grammar: ['verb tenses', 'subject-verb agreement', 'articles', 'prepositions']
- Vocabulary: ['word choice', 'phrasal verbs', 'collocations', 'academic vocabulary']
- Pronunciation: ['word stress', 'intonation', 'consonant sounds', 'vowel sounds']
- Fluency: ['speaking speed', 'hesitation', 'filler words', 'sentence linking']

Input sentence: {input}

Instructions:
- Identify the **category** and **sub-category** of the mistake as an English expert.
- Output **only** the category and sub-category in the following format: 

Mistake Category: [Category]  
Sub-category: [Sub-category]

### Example:

Input sentence: "She have many friends."

Expected output:
Mistake Category: Grammar  
Sub-category: subject-verb agreement
'''

        self.correct_sentence_template = '''
Imagine you are an expert in English, and your job is to correct mistakes in sentences. Given the following sentence with a mistake, provide the corrected version.

Input sentence: {input}

Instructions:
- Correct any grammar, vocabulary, pronunciation (if applicable), or fluency-related mistakes in the sentence.
- Output **only** the corrected sentence.

### Example:

Input sentence: "She have many friends."

Expected output:
"She has many friends."
'''


        # Initialize prompt templates
        self.classifier_prompt_template = PromptTemplate.from_template(self.classifier_template)
        self.correct_sentence_prompt_template = PromptTemplate.from_template(self.correct_sentence_template)

    # Method to get classifier response
    def get_classifier_response(self, user_input: str):
        prompt = self.classifier_prompt_template.format(input=user_input)
        response = self.llm.invoke(prompt)
        return response.content

    # Method to get corrected response
    def get_corrected_response(self, user_input: str):
        prompt = self.correct_sentence_prompt_template.format(input=user_input)
        response = self.llm.invoke(prompt)
        return response.content

class Recommender:
    def __init__(self,classifier_result, vectorstore, api_key: str = api_key, model_name: str = model_name):
        
        # Configure API key
        self.api_key = api_key
        self.vectorstore = vectorstore
        genai.configure(api_key=self.api_key)
        
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = self.api_key
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(model=model_name,temperature=0.0, google_api_key=self.api_key, max_tokens=None)

        self.d = self.vectorstore.similarity_search(classifier_result,k=1)                       # Example to retrive the contents using the Similarity search
        self.exer = self.d[0].page_content

        self.exercise_recommender_template = '''
Imagine you are an expert in language learning, and your job is to recommend exercises based on a user's demographics, mistake history, and preferred exercise type.

User demographics:
{user_demographic}

Mistake history:
{Mistake_history}

preferred exercise type:
{reco_exercise}


Instructions to follow:
- Based on the user's demographics, mistake history, and preferred exercise type, only 1 exercise.
- Output the exercise recommendation in a clear, concise format.
- Output only the exercise 
- Don't use any special symbols

Here is an example case:
Given the user demographics,history and exercise
Expected output:
Naruto is ___ ninja who dreams of becoming Hokage.
Luffy from "One Piece" set out to become ___ king of ___ pirates.
After watching ___ episode of "Attack on Titan," I felt ___ sense of excitement.
'''
      
        self.exercise_recommender_prompt_template = PromptTemplate.from_template(self.exercise_recommender_template)

    
    def get_exercise(self, user_ud: str, user_hist: str):
        prompt = self.exercise_recommender_prompt_template.format(
            user_demographic=user_ud, 
            Mistake_history=user_hist, 
            reco_exercise=self.exer
        )
        response = self.llm.invoke(prompt)
        return response.content
    

# Load documents
loader = DirectoryLoader(path="Exercises/", glob="**/*.txt")
pages = loader.load()

# Embedding model and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(
            documents=pages, 
            embedding=embeddings
)

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
