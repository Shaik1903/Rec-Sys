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


key = "AIzaSyAHPpqUignpGcTI1ZfmXfcFcxlpKDtDSrQ"
genai.configure(api_key=key)

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = key

# model = genai.GenerativeModel(model_name = "gemini-1.5-pro")
# response = model.generate_content("who is twitter ceo. Give me only the name")
# print(response.text)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",google_api_key=key)
loader = DirectoryLoader(path="Exercises/", glob="**/*.txt")
pages = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=pages, 
                                    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
# print(vectorstore._collection.count()) 

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.0, max_tokens=None)

classifier_template = '''
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

Category: [Category]  
Sub-category: [Sub-category]

### Example:

Input sentence: "She have many friends."

Expected output:
Category: Grammar  
Sub-category: subject-verb agreement


'''


classifier_prompt_template = PromptTemplate.from_template(classifier_template)

question = ""

def get_classifier_response(user_input):
    prompt = classifier_prompt_template.format(input=user_input)
    response = llm.invoke(prompt)
    return response

# question = "I think it's a good idea, um..."
# classifier_result = get_classifier_response(question)
# cls_res = classifier_result.content
# print(cls_res)


correct_sentence_template = '''
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

correct_sentence_prompt_template = PromptTemplate.from_template(correct_sentence_template)

def get_corrected_response(user_input):
    prompt = correct_sentence_prompt_template.format(input=user_input)
    response = llm.invoke(prompt)
    return response

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

exercise_recommender_template = '''
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

Here is an example case:
Given the user demographics,history and exercise
Expected output:
Naruto is ___ ninja who dreams of becoming Hokage.
Luffy from "One Piece" set out to become ___ king of ___ pirates.
After watching ___ episode of "Attack on Titan," I felt ___ sense of excitement.
'''

exercise_recommender_prompt_template = PromptTemplate.from_template(exercise_recommender_template)

question = ""

def get_exercise(user_ud,user_hist,exer):
    prompt = exercise_recommender_prompt_template.format(user_demographic=user_ud,Mistake_history = user_hist,reco_exercise = exer)
    response = llm.invoke(prompt)
    return response
