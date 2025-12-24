import re
import random
from fuzzywuzzy import fuzz, process
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class MedicinalPlantChatbot:
    """
    A rule-based chatbot for medicinal plant information
    Uses NLP and fuzzy matching for intelligent responses
    Now with conversational context and chunked information delivery
    """

    def __init__(self, plant_database):
        """
        Initialize chatbot with plant database

        Args:
            plant_database: Dictionary of plant information from PLANT_INFO
        """
        self.plant_db = plant_database
        self.plant_names = list(plant_database.keys())

        # Try to use NLTK stopwords, fallback if not available
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}

        # Conversation history
        self.conversation_history = []

        # Context tracking for multi-turn conversations
        self.current_plant = None
        self.current_topic = None
        self.info_sections_shown = set()

        # Initialize intents and responses
        self._setup_intents()

    def _setup_intents(self):
        """Setup conversation intents and patterns"""

        # Greetings
        self.greetings = {
            'patterns': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
            'responses': [
                "Hello! 🌿 I'm your medicinal plant assistant. How can I help you today?",
                "Hi there! 🌱 Ask me anything about medicinal plants!",
                "Greetings! 🍃 I can help you learn about 30 medicinal plants. What would you like to know?",
                "Hello! 👋 I'm here to help with medicinal plant information. What's your question?"
            ]
        }

        # Farewell
        self.farewells = {
            'patterns': ['bye', 'goodbye', 'see you', 'farewell', 'exit', 'quit'],
            'responses': [
                "Goodbye! 🌿 Stay healthy and take care!",
                "Farewell! 🌱 Feel free to come back anytime!",
                "Bye! 🍃 Hope I was helpful. Take care!",
                "See you later! 🌿 Wishing you good health!"
            ]
        }

        # Thanks
        self.thanks = {
            'patterns': ['thank', 'thanks', 'appreciate', 'grateful'],
            'responses': [
                "You're welcome! 🌿 Happy to help!",
                "My pleasure! 🌱 Feel free to ask more questions!",
                "Glad I could help! 🍃 Anything else you'd like to know?",
                "You're welcome! 😊 Don't hesitate to ask more!"
            ]
        }

        # Symptom to plant mapping
        self.symptom_map = {
            'diabetes': ['Curry', 'Drumstick', 'Fenugreek', 'Jamun', 'Neem', 'Tulsi'],
            'blood sugar': ['Curry', 'Fenugreek', 'Jamun', 'Neem'],
            'digestion': ['Basale', 'Betel', 'Curry', 'Fenugreek', 'Guava', 'Mint', 'Mexican_Mint'],
            'digestive': ['Basale', 'Betel', 'Curry', 'Fenugreek', 'Guava', 'Mint', 'Mexican_Mint'],
            'stomach': ['Betel', 'Guava', 'Mint', 'Fenugreek'],
            'cough': ['Mexican_Mint', 'Tulsi', 'Betel'],
            'cold': ['Mexican_Mint', 'Tulsi', 'Mint', 'Betel'],
            'fever': ['Parijata', 'Tulsi', 'Neem', 'Guava'],
            'skin': ['Neem', 'Sandalwood', 'Tulsi', 'Hibiscus', 'Indian_Beech'],
            'hair': ['Curry', 'Hibiscus', 'Fenugreek'],
            'heart': ['Pomegranate', 'Drumstick', 'Guava'],
            'blood pressure': ['Hibiscus', 'Pomegranate', 'Curry'],
            'immunity': ['Drumstick', 'Guava', 'Lemon', 'Tulsi', 'Neem'],
            'anemia': ['Arive-Dantu', 'Karanda'],
            'wound': ['Crape_Jasmine', 'Neem', 'Betel', 'Peepal'],
            'headache': ['Betel', 'Mint', 'Jamaica_Cherry-Gasagase'],
            'pain': ['Parijata', 'Rasna', 'Indian_Mustard', 'Jamaica_Cherry-Gasagase'],
            'joint': ['Parijata', 'Rasna', 'Indian_Beech'],
            'arthritis': ['Parijata', 'Rasna'],
            'respiratory': ['Mexican_Mint', 'Tulsi', 'Peepal'],
            'stress': ['Jasmine', 'Tulsi', 'Parijata'],
            'anxiety': ['Jasmine', 'Tulsi', 'Parijata'],
            'weight loss': ['Lemon', 'Guava']
        }

    def preprocess_text(self, text):
        """Clean and preprocess user input"""
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def extract_keywords(self, text):
        """Extract important keywords from text"""
        words = text.split()
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        return keywords

    def find_plant_name(self, text):
        """
        Find plant name in user query using fuzzy matching
        Returns: (plant_name, confidence_score)
        """
        text = self.preprocess_text(text)

        # Try exact match first
        for plant in self.plant_names:
            if plant.lower().replace('_', ' ') in text or plant.lower().replace('-', ' ') in text:
                return plant, 100

        # Try fuzzy matching
        best_match = process.extractOne(text, self.plant_names, scorer=fuzz.token_set_ratio)

        if best_match and best_match[1] > 60:
            return best_match[0], best_match[1]

        return None, 0

    def detect_intent(self, user_input):
        """Detect user's intent from input"""
        text = self.preprocess_text(user_input)

        # Check for "more" or "tell me more" - continuation intent
        if any(pattern in text for pattern in ['more', 'tell me more', 'what else', 'continue', 'next']):
            return 'more_info'

        # Check greetings
        if any(pattern in text for pattern in self.greetings['patterns']):
            return 'greeting'

        # Check farewells
        if any(pattern in text for pattern in self.farewells['patterns']):
            return 'farewell'

        # Check thanks
        if any(pattern in text for pattern in self.thanks['patterns']):
            return 'thanks'

        # Check for plant name query
        plant, confidence = self.find_plant_name(text)
        if plant:
            return 'plant_info'

        # Check for symptom query
        if any(keyword in text for keyword in ['cure', 'treat', 'help', 'good for', 'remedy', 'medicine', 'heal']):
            return 'symptom_query'

        # Check for specific info requests
        if any(keyword in text for keyword in ['benefit', 'use', 'medicinal']):
            return 'benefits_query'

        if any(keyword in text for keyword in ['prepare', 'how to', 'consume', 'take', 'make']):
            return 'preparation_query'

        if any(keyword in text for keyword in ['precaution', 'warning', 'side effect', 'safe']):
            return 'precautions_query'

        if any(keyword in text for keyword in ['part', 'which part']):
            return 'parts_query'

        # Check for list query
        if any(keyword in text for keyword in ['list', 'all plants', 'what plants', 'available', 'show me']):
            return 'list_plants'

        # Check for help
        if any(keyword in text for keyword in ['help', 'what can you do', 'guide']):
            return 'help'

        return 'general'

    def get_plant_brief_intro(self, plant):
        """Get a brief introduction to a plant"""
        info = self.plant_db[plant]
        display_name = plant.replace('_', ' ').replace('-', ' ')

        response = f"🌿 **{display_name}**\n\n"
        response += f"*{info['scientific_name']}* ({info['family']} family)\n\n"

        # Extract first sentence or first 150 chars of description
        desc = info['description']
        first_sentence = desc.split('.')[0] + '.'
        if len(first_sentence) > 150:
            first_sentence = desc[:150] + '...'

        response += f"{first_sentence}\n\n"
        response += f"**Also known as:** {info['common_names']}"

        return response

    def get_plant_benefits(self, plant):
        """Get medicinal benefits of a plant in a digestible format"""
        info = self.plant_db[plant]
        display_name = plant.replace('_', ' ').replace('-', ' ')

        uses = info['uses']

        # Try to split uses into bullet points if they contain common separators
        if '.' in uses and len(uses) > 200:
            # Split by sentences and take first 3-4 key benefits
            sentences = [s.strip() for s in uses.split('.') if s.strip()]
            key_benefits = sentences[:3]

            response = f"🌿 **{display_name} - Key Medicinal Benefits:**\n\n"
            for i, benefit in enumerate(key_benefits, 1):
                response += f"{i}. {benefit}\n\n"
        else:
            # Shorter uses - show all
            response = f"🌿 **{display_name} - Medicinal Uses:**\n\n{uses}\n\n"
            response += f"**Parts Used:** {info['parts_used']}"

        return response

    def get_plant_preparation(self, plant):
        """Get preparation methods for a plant"""
        info = self.plant_db[plant]
        display_name = plant.replace('_', ' ').replace('-', ' ')

        response = f"🧪 **How to Prepare {display_name}:**\n\n"
        response += f"{info['preparation']}\n\n"
        response += f"**Parts to Use:** {info['parts_used']}"

        return response

    def get_plant_precautions(self, plant):
        """Get precautions for a plant"""
        info = self.plant_db[plant]
        display_name = plant.replace('_', ' ').replace('-', ' ')

        response = f"⚠️ **{display_name} - Safety Precautions:**\n\n"
        response += f"{info['precautions']}"

        return response

    def get_plant_complete_info(self, plant):
        """Get complete information about a plant"""
        info = self.plant_db[plant]
        display_name = plant.replace('_', ' ').replace('-', ' ')

        response = f"🌿 **{display_name} - Complete Information**\n\n"
        response += f"**Scientific Name:** *{info['scientific_name']}*\n\n"
        response += f"**Common Names:** {info['common_names']}\n\n"
        response += f"**Family:** {info['family']}\n\n"
        response += f"**Description:** {info['description']}\n\n"
        response += f"**Medicinal Uses:** {info['uses']}\n\n"
        response += f"**Parts Used:** {info['parts_used']}\n\n"
        response += f"**Preparation:** {info['preparation']}\n\n"
        response += f"**⚠️ Precautions:** {info['precautions']}\n\n"

        return response

    def handle_more_info(self):
        """Handle requests for more information about current plant"""
        if not self.current_plant:
            return "What plant would you like to know more about? Type a plant name or 'list' to see all plants."

        # Show complete info if they want more
        return self.get_plant_complete_info(self.current_plant)

    def get_plant_info(self, user_input):
        """Get information about a specific plant"""
        plant, confidence = self.find_plant_name(user_input)

        if not plant:
            return "I couldn't identify the plant you're asking about. Could you please spell it more clearly? Or type 'list' to see all available plants."

        # Update context
        self.current_plant = plant
        self.info_sections_shown = set()

        # Check what specific info they want
        text = self.preprocess_text(user_input)

        if any(word in text for word in ['benefit', 'use', 'medicinal', 'property']):
            self.info_sections_shown.add('benefits')
            return self.get_plant_benefits(plant)
        elif any(word in text for word in ['prepare', 'how', 'consume', 'take', 'make']):
            self.info_sections_shown.add('preparation')
            return self.get_plant_preparation(plant)
        elif any(word in text for word in ['precaution', 'warning', 'side effect', 'safe']):
            self.info_sections_shown.add('precautions')
            return self.get_plant_precautions(plant)
        else:
            # Default: show brief intro
            return self.get_plant_brief_intro(plant)

    def find_plants_for_symptom(self, user_input):
        """Find plants that can help with a symptom"""
        text = self.preprocess_text(user_input)
        keywords = self.extract_keywords(text)

        recommended_plants = set()

        for symptom, plants in self.symptom_map.items():
            if symptom in text or any(symptom in keyword for keyword in keywords):
                recommended_plants.update(plants)

        if recommended_plants:
            # Limit to top 5 plants initially
            plants_list = sorted(list(recommended_plants))[:5]

            response = "🌿 **Here are some plants that may help:**\n\n"
            for i, plant in enumerate(plants_list, 1):
                if plant in self.plant_db:
                    display_name = plant.replace('_', ' ')
                    # Show just first 80 chars of uses
                    uses_preview = self.plant_db[plant]['uses'][:80] + '...'
                    response += f"{i}. **{display_name}** - {uses_preview}\n\n"

            response += "\n💡 *Type any plant name to learn more about it!*\n"

            if len(recommended_plants) > 5:
                response += f"\n*({len(recommended_plants) - 5} more plants available - ask me for alternatives)*"

            return response
        else:
            return "I couldn't find specific plants for that symptom. Could you rephrase your query? For example: 'What is good for diabetes?' or 'Plants for cough'"

    def list_all_plants(self):
        """List all available plants in a compact format"""
        response = "🌿 **Available Medicinal Plants (30):**\n\n"

        plants_sorted = sorted(self.plant_names)

        # Show in columns for better readability
        for i in range(0, len(plants_sorted), 3):
            row_plants = plants_sorted[i:i + 3]
            row = ' | '.join([p.replace('_', ' ') for p in row_plants])
            response += f"{row}\n"

        response += "\n💡 *Type any plant name to get information!*"
        return response

    def get_help(self):
        """Provide help information"""
        return """🤖 **How I Can Help You:**

1️⃣ **Ask about a specific plant:**
   - "Tell me about Tulsi"
   - "What is Neem?"
   - "Benefits of Curry leaves"

2️⃣ **Find plants for health issues:**
   - "What is good for diabetes?"
   - "Plants for cough"
   - "Remedy for skin problems"

3️⃣ **Get specific information:**
   - "How to prepare Fenugreek?"
   - "Precautions for Neem"
   - Type 'more' for complete details

4️⃣ **List all plants:**
   - "Show me all plants"
   - "List available plants"

💡 **I'll give you brief, easy-to-read information. Ask for 'more' whenever you want complete details!**"""

    def chat(self, user_input):
        """
        Main chat function - process user input and return response

        Args:
            user_input: User's message

        Returns:
            Bot's response string
        """
        self.conversation_history.append({'user': user_input})

        intent = self.detect_intent(user_input)

        if intent == 'greeting':
            response = random.choice(self.greetings['responses'])
            self.current_plant = None

        elif intent == 'farewell':
            response = random.choice(self.farewells['responses'])
            self.current_plant = None

        elif intent == 'thanks':
            response = random.choice(self.thanks['responses'])

        elif intent == 'more_info':
            response = self.handle_more_info()

        elif intent == 'plant_info':
            response = self.get_plant_info(user_input)

        elif intent == 'benefits_query':
            if self.current_plant:
                response = self.get_plant_benefits(self.current_plant)
            else:
                plant, confidence = self.find_plant_name(user_input)
                if plant:
                    self.current_plant = plant
                    response = self.get_plant_benefits(plant)
                else:
                    response = "Which plant's benefits would you like to know about? Type a plant name or 'list' to see all plants."

        elif intent == 'preparation_query':
            if self.current_plant:
                response = self.get_plant_preparation(self.current_plant)
            else:
                plant, confidence = self.find_plant_name(user_input)
                if plant:
                    self.current_plant = plant
                    response = self.get_plant_preparation(plant)
                else:
                    response = "Which plant would you like to know how to prepare? Type a plant name or 'list' to see all plants."

        elif intent == 'precautions_query':
            if self.current_plant:
                response = self.get_plant_precautions(self.current_plant)
            else:
                plant, confidence = self.find_plant_name(user_input)
                if plant:
                    self.current_plant = plant
                    response = self.get_plant_precautions(plant)
                else:
                    response = "Which plant's precautions would you like to know? Type a plant name or 'list' to see all plants."

        elif intent == 'symptom_query':
            response = self.find_plants_for_symptom(user_input)

        elif intent == 'list_plants':
            response = self.list_all_plants()
            self.current_plant = None

        elif intent == 'help':
            response = self.get_help()

        else:
            response = """I'm not sure I understood that. Here's what I can help you with:

🌿 Ask about specific plants (e.g., "Tell me about Neem")
💊 Find remedies for health issues (e.g., "What helps with diabetes?")
📋 List all available plants (type "list")
❓ Get help (type "help")

What would you like to know?"""
            self.current_plant = None

        self.conversation_history.append({'bot': response})

        return response

    def get_conversation_history(self):
        """Return conversation history"""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.current_plant = None
        self.info_sections_shown = set()