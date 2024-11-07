from cairosvg import svg2png
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests

load_dotenv()

# Initialize the OpenAI client with an API key.
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



def run_function(name: str, args: dict):
    if name == "svg_to_png_bytes":
        return svg_to_png_bytes(args["svg_string"])
    if name == "generate_image":
        return generate_image(args["prompt"])
    return None

def generate_image(prompt: str):
    response = openai.images.generate(
        prompt=prompt, model="dall-e-3", n=1, size="1024x1024"
    )
    image_url = response.data[0].url
    image_response = requests.get(image_url)
    return image_response

def svg_to_png_bytes(svg_string):
    # Convert SVG string to PNG bytes
    png_bytes = svg2png(bytestring=svg_string.encode("utf-8"))
    return png_bytes


functions = [
    {
        "name": "svg_to_png_bytes",
        "description": "Generate a PNG from an SVG",
        "parameters": {
            "type": "object",
            "properties": {
                "svg_string": {
                    "type": "string",
                    "description": "A fully formed SVG element in the form of a string",
                },
            },
            "required": ["svg_string"],
        },
    },
    {
        "name": "generate_image",
        "description": "generate an image using the dalle api from a prompt",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "an image generation prompt",
                },
            },
            "required": ["prompt"],
        },
    },
]
