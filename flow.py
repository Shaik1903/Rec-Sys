import os
from typing import List, Dict, Tuple

class ExerciseRecommender:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.mistake_categories = {
            'grammar': ['verb tenses', 'subject-verb agreement', 'articles', 'prepositions'],
            'vocabulary': ['word choice', 'phrasal verbs', 'collocations', 'academic vocabulary'],
            'pronunciation': ['word stress', 'intonation', 'consonant sounds', 'vowel sounds'],
            'fluency': ['speaking speed', 'hesitation', 'filler words', 'sentence linking']
        }
        self.exercise_database = self._initialize_exercise_database()

    

    def classify_input(self, user_input: str, user_history: List[Dict]) -> Tuple[str, float]:
        prompt = f"""
        Based on the following user input and their history of mistakes, classify the category 
        of potential errors. User input: {user_input}
        
        User history of mistakes:
        {self._format_history(user_history)}
        
        Available categories: {', '.join(self.mistake_categories.keys())}
        
        Analyze the input for:
        1. Grammar issues (verb tenses, subject-verb agreement, articles, prepositions)
        2. Vocabulary problems (incorrect word choice, phrasal verbs, collocations)
        3. Pronunciation concerns (stress, intonation, specific sound issues)
        4. Fluency challenges (speaking speed, hesitation, filler words, sentence connecting)
        
        Return format: (category, confidence_score)
        """
        
        # Simulated LLM call - replace with actual API call
        # response = call_llm_api(prompt)
        category, confidence = self._mock_llm_classification(user_input)
        return category, confidence

    def _format_history(self, history: List[Dict]) -> str:
        formatted_history = ""
        for entry in history:
            formatted_history += f"- Mistake: {entry['mistake']}, Category: {entry['category']}\n"
        return formatted_history

    

    def recommend_exercises(self, category: str, confidence: float, difficulty: str = 'intermediate') -> List[Dict]:
        available_exercises = self.exercise_database.get(category, [])
        
        # Filter exercises based on difficulty
        recommended = [ex for ex in available_exercises if ex['difficulty'] == difficulty]
        
        # If no exercises match the difficulty, return all exercises in the category
        if not recommended:
            recommended = available_exercises
        
        return recommended

    def get_personalized_recommendation(self, user_input: str, user_history: List[Dict]) -> Dict:
        category, confidence = self.classify_input(user_input, user_history)
        exercises = self.recommend_exercises(category, confidence)
        
        return {
            'category': category,
            'confidence': confidence,
            'recommended_exercises': exercises
        }

# Example usage
def main():
    # Sample user history
    user_history = [
        {'mistake': 'Incorrect verb tense in conditional sentence', 'category': 'grammar'},
        {'mistake': 'Mispronounced "th" sound consistently', 'category': 'pronunciation'},
        {'mistake': 'Used too many filler words like "um" and "uh"', 'category': 'fluency'}
    ]
    
    # Initialize recommender
    recommender = ExerciseRecommender(api_key='your-api-key-here')
    
    # Sample user inputs to test different categories
    test_inputs = [
        "I having trouble with past tense verbs",
        "I don't know how to pronounce 'th' sound correctly",
        "I need more words to express my ideas",
        "I speak too slow and use many um and ah"
    ]
    
    # Test each input
    for user_input in test_inputs:
        print(f"\nTesting input: '{user_input}'")
        recommendation = recommender.get_personalized_recommendation(user_input, user_history)
        
        print(f"Classified Category: {recommendation['category']}")
        print(f"Confidence Score: {recommendation['confidence']}")
        print("Recommended Exercises:")
        for exercise in recommendation['recommended_exercises']:
            print(f"- {exercise['name']} ({exercise['difficulty']})")
            print(f"  {exercise['description']}")

if __name__ == "__main__":
    main()