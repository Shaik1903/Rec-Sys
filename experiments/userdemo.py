import os
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
import random

@dataclass
class UserProfile:
    native_language: str
    interests: List[str]
    proficiency_level: str
    learning_history: Dict[str, float] = None  # Error type to success rate mapping

    def __post_init__(self):
        if self.learning_history is None:
            self.learning_history = {}

@dataclass
class ExerciseAttempt:
    exercise_id: str
    user_answers: List[str]
    correct_answers: List[str]
    time_taken: int  # in seconds
    difficulty: str
    error_type: str

class AdaptiveExerciseGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.exercise_history = {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_openai_api(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an adaptive language teacher who creates personalized exercises."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise

    def generate_exercise(self, error_type: str, user_profile: UserProfile) -> Dict:
        context = self._get_personalized_context(user_profile)
        difficulty = self._determine_difficulty(error_type, user_profile)
        
        prompt = f"""
        Create a language exercise with the following requirements:

        1. Error focus: {error_type}
        2. Context: Use elements from {context}
        3. Difficulty: {difficulty}
        4. Format: Create exactly 5 fill-in-the-blank or multiple choice questions

        Return the exercise in this JSON format:
        {{
            "context": "brief text explaining the theme",
            "questions": [
                {{
                    "question": "The actual question with ___ for blanks",
                    "correct_answer": "the right answer",
                    "options": ["option1", "option2", "option3", "option4"] // if multiple choice
                }}
            ],
            "difficulty": "{difficulty}",
            "theme": "what interest/cultural element this exercise uses"
        }}
        """
        
        try:
            response = self._call_openai_api(prompt)
            exercise = json.loads(response)
            exercise_id = f"ex_{random.randint(10000, 99999)}"
            self.exercise_history[exercise_id] = exercise
            exercise['id'] = exercise_id
            return exercise
        except json.JSONDecodeError:
            return self._generate_fallback_exercise(error_type, difficulty, context)

    def _get_personalized_context(self, user_profile: UserProfile) -> str:
        cultural_elements = self._get_cultural_references(user_profile.native_language)
        interests = ', '.join(user_profile.interests)
        return f"{cultural_elements} and user interests: {interests}"

    def _get_cultural_references(self, native_language: str) -> str:
        cultural_mapping = {
            "Japanese": "anime, manga, J-pop culture, Japanese cuisine",
            "Korean": "K-pop, Korean dramas, Korean street food, PC bang gaming",
            "Spanish": "soccer/football, telenovelas, regional cuisine, local festivals",
            # Add more mappings as needed
        }
        return cultural_mapping.get(native_language, "general global popular culture")

    def _determine_difficulty(self, error_type: str, user_profile: UserProfile) -> str:
        success_rate = user_profile.learning_history.get(error_type, 0.5)
        if success_rate < 0.4:
            return 'beginner'
        elif success_rate < 0.7:
            return 'intermediate'
        return 'advanced'

    def score_attempt(self, attempt: ExerciseAttempt) -> Tuple[float, Dict]:
        exercise = self.exercise_history.get(attempt.exercise_id)
        if not exercise:
            return 0.0, {"error": "Exercise not found"}

        questions = exercise['questions']
        correct_count = sum(1 for ua, ca in zip(attempt.user_answers, attempt.correct_answers) if ua == ca)
        score = correct_count / len(questions)

        analysis = {
            "score": score,
            "time_per_question": attempt.time_taken / len(questions),
            "question_breakdown": [
                {
                    "question": q['question'],
                    "user_answer": ua,
                    "correct_answer": ca,
                    "is_correct": ua == ca
                }
                for q, ua, ca in zip(questions, attempt.user_answers, attempt.correct_answers)
            ]
        }

        return score, analysis

    def update_user_profile(self, user_profile: UserProfile, attempt: ExerciseAttempt, score: float):
        # Update learning history with exponential moving average
        alpha = 0.3  # learning rate
        current = user_profile.learning_history.get(attempt.error_type, 0.5)
        user_profile.learning_history[attempt.error_type] = (alpha * score) + ((1 - alpha) * current)

    def _generate_fallback_exercise(self, error_type: str, difficulty: str, context: str) -> Dict:
        return {
            "id": f"ex_{random.randint(10000, 99999)}",
            "context": f"Practice {error_type} using {context}",
            "questions": [
                {
                    "question": f"Simple {error_type} practice question ___.",
                    "correct_answer": "answer",
                    "options": ["answer", "wrong1", "wrong2", "wrong3"]
                }
            ],
            "difficulty": difficulty,
            "theme": "general practice"
        }

def main():
    api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
    generator = AdaptiveExerciseGenerator(api_key=api_key)
    
    # Sample user profile
    user_profile = UserProfile(
        native_language="Japanese",
        interests=["anime", "video games", "cooking"],
        proficiency_level="intermediate",
        learning_history={"prepositions": 0.4, "articles": 0.7}
    )
    
    # Generate and test an exercise
    exercise = generator.generate_exercise("prepositions", user_profile)
    print("\nGenerated Exercise:")
    print(f"Theme: {exercise['theme']}")
    print(f"Context: {exercise['context']}")
    print("\nQuestions:")
    for i, q in enumerate(exercise['questions'], 1):
        print(f"{i}. {q['question']}")
        if 'options' in q:
            print(f"   Options: {', '.join(q['options'])}")
    
    # Simulate user attempt
    simulated_attempt = ExerciseAttempt(
        exercise_id=exercise['id'],
        user_answers=["answer1", "answer2", "answer3", "answer4", "answer5"],
        correct_answers=[q['correct_answer'] for q in exercise['questions']],
        time_taken=120,
        difficulty=exercise['difficulty'],
        error_type="prepositions"
    )
    
    # Score attempt
    score, analysis = generator.score_attempt(simulated_attempt)
    
    print("\nAttempt Analysis:")
    print(f"Score: {score * 100:.1f}%")
    print(f"Average time per question: {analysis['time_per_question']:.1f} seconds")
    
    # Update user profile
    generator.update_user_profile(user_profile, simulated_attempt, score)
    
    print("\nUpdated User Profile:")
    for error_type, success_rate in user_profile.learning_history.items():
        print(f"{error_type}: {success_rate:.2f} success rate")

if __name__ == "__main__":
    main()