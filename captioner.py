import random
import os
import json
import numpy as np
from ollama import Client
import httpx
from PIL import Image
from io import BytesIO
from typing import List, Any

def load_config(key: str) -> Any:
    config_path = os.path.join(os.path.dirname(__file__), "etc/config.json")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get(key, "")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Failed to load {key} from config - {e}")
        return ""

def get_ollama_models() -> List[str]:
    return load_config("ollama_models")

def get_ollama_url() -> str:
    return load_config("ollama_url")

class CaptionerNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        ollama_models = get_ollama_models()

        return {
            "required": {
                "images": ("IMAGE",),
                "query": ("STRING", {
                    "multiline": True,
                    "default": "Please describe the image as detailed as possible. If there are celebrities, movie characters, famous attractions or well-known things in the image, please use their names directly. No more than 200 words"
                }),
                "ollama_model": (ollama_models, {"default": ollama_models[0]}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "ollama_captioner"
    CATEGORY = "Huixingyun"

    def ollama_captioner(self, images, query, ollama_model, seed):
        images_binary = []

        for (batch_number, image) in enumerate(images):
            # Convert tensor to numpy array
            i = 255. * image.cpu().numpy()
            
            # Create PIL Image
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Save to BytesIO buffer
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            
            # Get binary data
            img_binary = buffered.getvalue()
            images_binary.append(img_binary)
        
        options = {"seed": seed,}

        client = Client(host=get_ollama_url(), timeout=httpx.Timeout(60.0))
        response = client.generate(
            model   = ollama_model, 
            prompt  = query, 
            images  = images_binary,
            format  = '',
            options = options,
        )

        return (response['response'],)


NODE_CLASS_MAPPINGS = {
    "HXOllamaCaptioner": CaptionerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HXOllamaCaptioner": "HX Ollama Captioner"
}
