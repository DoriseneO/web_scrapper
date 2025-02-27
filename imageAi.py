import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Interactive Loop for Image Generation
while True:
    prompt = input("\nDescribe the image you want to generate (or type 'exit' to quit): ")
    
    if prompt.lower() == "exit":
        print("👋 Exiting. Have a great day!")
        break

    try:
        # Generate an image using DALL·E 3
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1  # Generate 1 image
        )
        
        # Extract image URL
        image_url = response.data[0].url
        print("\n🖼️ Generated Image URL:", image_url)

    except Exception as e:
        print("\n❌ Error:", e)
