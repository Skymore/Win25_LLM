import openai
from openai import OpenAI
from PIL import Image
import urllib.request
from io import BytesIO
from IPython.display import display

class open_ai_a0:
    def __init__(self):
        api_key = self.load_key()
        self.client = OpenAI(api_key=api_key)

    # Loading OpenAI key from file #
    def load_key(self):
        open_ai_key_file = "open_ai_key.txt"
        with open(open_ai_key_file, "r") as f:
            for line in f:
                OPENAI_KEY = line.strip()
                return OPENAI_KEY

    def get_sentiment(self, text):
        # Prompt for sentiment analysis
        prompt = f"""Analyze the sentiment of the following text and classify it as POSITIVE, NEGATIVE, or NEUTRAL. 
        Then provide a brief explanation of why.
        
        Text: {text}
        
        Format your response as:
        Sentiment: [POSITIVE/NEGATIVE/NEUTRAL]
        Explanation: [Your explanation]"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=1,
        )

        sentiment = response.choices[0].message.content
        print(sentiment)

    def text_to_image(self):
        response = self.client.images.generate(
            model="dall-e-2",
            prompt="Create a vibrant and artistic visualization that represents the emotional tone of the analyzed text. Include elements that symbolize the sentiment detected.",
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        with urllib.request.urlopen(image_url) as image_url:
            img = Image.open(BytesIO(image_url.read()))

        display(img)

    def run(self):
        text = input("Enter a text: ")
        self.get_sentiment(text)
        self.text_to_image()

if __name__ == "__main__":
    llm = open_ai_a0()
    llm.run()
