import google.generativeai as genai

class Message:
    def __init__(self, content):
        self.content = content

class Gemini:
    def __init__(self, model: str, api_key: str):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.client = genai.GenerativeModel(model)

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=None,
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ):
        conversation = ""
        if system:
            conversation += f"System: {system}\n"
        if messages:
            for msg in messages:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                conversation += f"{role}: {content}\n"

        if tools:
            conversation += f"\n[Tools available: {tools}]"

        response = self.client.generate_content(conversation)
        return Message(response.text)

    def add_assistant_message(self, messages, content):
        messages.append({
            "role": "assistant",
            "content": content
        })
    def text_from_message(self, message):
        if isinstance(message, Message):
            return message.content
        if isinstance(message, dict) and "content" in message:
            return message["content"]
        return str(message)
