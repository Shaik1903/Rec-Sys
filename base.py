import os
from typing import List, Dict, Tuple

class ExerciseRecommender:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.mistake_categories = {
            'grammar': ['subject-verb agreement', 'tense errors', 'article usage'],
            'vocabulary': ['word choice', 'idiom usage', 'collocations'],
            'pronunciation': ['consonant sounds', 'vowel sounds', 'stress patterns'],
            'sentence_structure': ['word order', 'complex sentences', 'run-on sentences']
        }
        self.exercise_database = self._initialize_exercise_database()

    def _initialize_exercise_database(self) -> Dict[str, List[Dict]]:
        return {
            'grammar': [
                {'name': 'Subject-Verb Quiz', 'difficulty': 'beginner', 'focus': 'subject-verb agreement'},
                {'name': 'Tense Timeline', 'difficulty': 'intermediate', 'focus': 'tense errors'},
                {'name': 'Article Adventure', 'difficulty': 'advanced', 'focus': 'article usage'}
            ],
            'vocabulary': [
                {'name': 'Word Choice Challenge', 'difficulty': 'beginner', 'focus': 'word choice'},
                {'name': 'Idiom Mastery', 'difficulty': 'intermediate', 'focus': 'idiom usage'},
                {'name': 'Collocation Connection', 'difficulty': 'advanced', 'focus': 'collocations'}
            ],
            # Add more categories as needed
        }

    def classify_input(self, user_input: str, user_history: List[Dict]) -> Tuple[str, float]:
        prompt = f"""
        Based on the following user input and their history of mistakes, classify the category 
        of potential errors. User input: {user_input}
        
        User history of mistakes:
        {self._format_history(user_history)}
        
        Available categories: {', '.join(self.mistake_categories.keys())}
        
        Return format: (category, confidence_score)
        """
        
        # Simulated LLM call - replace with actual API call
        # response = call_llm_api(prompt)
        # For demonstration, we'll return a mock response
        category, confidence = self._mock_llm_classification(user_input)
        return category, confidence

    def _format_history(self, history: List[Dict]) -> str:
        formatted_history = ""
        for entry in history:
            formatted_history += f"- Mistake: {entry['mistake']}, Category: {entry['category']}\n"
        return formatted_history

    def _mock_llm_classification(self, user_input: str) -> Tuple[str, float]:
        # This is a mock function - replace with actual LLM logic
        if 'verb' in user_input.lower():
            return 'grammar', 0.85
        elif any(word in user_input.lower() for word in ['mean', 'definition', 'word']):
            return 'vocabulary', 0.78
        return 'sentence_structure', 0.65

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
        {'mistake': 'Used "is" instead of "are" with plural noun', 'category': 'grammar'},
        {'mistake': 'Incorrect use of "make" vs "do"', 'category': 'vocabulary'},
    ]
    
    # Initialize recommender
    recommender = ExerciseRecommender(api_key='your-api-key-here')
    
    # Sample user input
    user_input = "I having trouble with verb tenses"
    
    # Get recommendation
    recommendation = recommender.get_personalized_recommendation(user_input, user_history)
    
    # Print results
    print(f"Category: {recommendation['category']}")
    print(f"Confidence: {recommendation['confidence']}")
    print("\nRecommended Exercises:")
    for exercise in recommendation['recommended_exercises']:
        print(f"- {exercise['name']} (Difficulty: {exercise['difficulty']})")

if __name__ == "__main__":
    main()