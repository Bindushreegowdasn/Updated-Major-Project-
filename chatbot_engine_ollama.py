import requests
import json


class MedicinalPlantChatbot:
    """
    Ollama-powered chatbot for medicinal plant information
    Uses Llama model for intelligent, context-aware responses
    """

    def __init__(self, plant_database, model="llama3.2"):
        """
        Initialize chatbot with plant database and Ollama model

        Args:
            plant_database: Dictionary of plant information from PLANT_INFO
            model: Ollama model name (default: llama3.2)
        """
        self.plant_db = plant_database
        self.plant_names = list(plant_database.keys())
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"

        # Conversation history
        self.conversation_history = []
        self.current_plant = None

        # Create knowledge base for the AI
        self.knowledge_base = self._build_knowledge_base()

    def _build_knowledge_base(self):
        """Build a comprehensive knowledge base string from plant database"""
        kb = "MEDICINAL PLANT DATABASE:\n\n"

        for plant_name, info in self.plant_db.items():
            display_name = plant_name.replace('_', ' ').replace('-', ' ')
            kb += f"PLANT: {display_name}\n"
            kb += f"Scientific Name: {info['scientific_name']}\n"
            kb += f"Common Names: {info['common_names']}\n"
            kb += f"Family: {info['family']}\n"
            kb += f"Description: {info['description']}\n"
            kb += f"Uses: {info['uses']}\n"
            kb += f"Parts Used: {info['parts_used']}\n"
            kb += f"Preparation: {info['preparation']}\n"
            kb += f"Precautions: {info['precautions']}\n\n"
            kb += "-" * 80 + "\n\n"

        return kb

    def set_current_plant(self, plant_name):
        """Set the current plant context for conversation"""
        self.current_plant = plant_name

    def _call_ollama(self, prompt, system_message=None):
        """
        Call Ollama API to generate response

        Args:
            prompt: User prompt
            system_message: Optional system message for context

        Returns:
            Generated response string
        """
        try:
            full_prompt = prompt
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt}"

            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            }

            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            return result.get('response', '').strip()

        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to Ollama. Make sure Ollama is running with 'ollama serve'")
        except requests.exceptions.Timeout:
            raise Exception("Ollama request timed out. The model might be loading.")
        except Exception as e:
            raise Exception(f"Ollama error: {str(e)}")

    def get_quick_plant_summary(self, plant_name):
        """
        Get a quick AI-generated summary of a plant

        Args:
            plant_name: Name of the plant

        Returns:
            Brief summary string
        """
        if plant_name not in self.plant_db:
            return f"Plant '{plant_name}' not found in database."

        info = self.plant_db[plant_name]
        display_name = plant_name.replace('_', ' ').replace('-', ' ')

        system_msg = """You are a helpful medicinal plant expert. Provide brief, informative summaries.
Keep responses concise (2-3 sentences) and engaging. Focus on the most interesting medicinal benefits."""

        prompt = f"""Provide a brief, engaging 2-3 sentence summary about {display_name} ({info['scientific_name']}).

Plant Information:
- Uses: {info['uses']}
- Description: {info['description']}

Create a friendly, informative summary highlighting its key medicinal benefits."""

        try:
            return self._call_ollama(prompt, system_msg)
        except Exception as e:
            # Fallback to basic info if Ollama fails
            return f"{display_name} is known for: {info['uses'][:100]}..."

    def chat(self, user_input):
        """
        Main chat function - process user input and return AI-generated response

        Args:
            user_input: User's message

        Returns:
            Bot's response string
        """
        # Don't add to history here - let the app handle it
        # Build context-aware system message
        system_msg = f"""You are an expert medicinal plant advisor with knowledge of 30 medicinal plants.

IMPORTANT GUIDELINES:
1. Provide accurate, helpful information about medicinal plants
2. Keep responses concise but informative (3-5 sentences unless more detail is requested)
3. Always mention safety precautions when relevant
4. Use emoji occasionally to make responses friendly 🌿
5. If asked about plants not in the database, politely say you specialize in these 30 plants
6. Be conversational and helpful
7. Never repeat the same information multiple times

{self.knowledge_base}

Current conversation context:
- Current plant being discussed: {self.current_plant if self.current_plant else 'None'}
- Conversation history: {len(self.conversation_history)} messages

Answer the user's question based on the plant database above."""

        # Add conversation context for continuity
        conversation_context = ""
        if len(self.conversation_history) > 1:
            recent_history = self.conversation_history[-4:]  # Last 2 exchanges
            conversation_context = "Recent conversation:\n"
            for msg in recent_history:
                role = "User" if msg['role'] == 'user' else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
            conversation_context += "\n"

        full_prompt = f"{conversation_context}User's current question: {user_input}"

        try:
            response = self._call_ollama(full_prompt, system_msg)

            # Add to internal history after getting response
            self.conversation_history.append({'role': 'user', 'content': user_input})
            self.conversation_history.append({'role': 'assistant', 'content': response})

            return response

        except Exception as e:
            error_msg = f"⚠️ Ollama Error: {str(e)}"
            return error_msg

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.current_plant = None

    def get_conversation_history(self):
        """Return conversation history"""
        return self.conversation_history