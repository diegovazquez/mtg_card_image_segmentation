'''
https://developers.cloudflare.com/workers-ai/models/stable-diffusion-v1-5-inpainting/
inpaint outpaint
'''
import os
import random
import requests
import base64
from typing import Union
from pathlib import Path
import io
from PIL import Image, ImageOps

def gen_prompt():
    base_prompts = [
        "A hand holding a card",
        "cards floating in mid-air",
        "a card whit dramatic shadows",
        "A fortune teller's hands hovering over tarot cards, crystal ball in background",
        "Fantasy card game battle scene",
        "Rare collectible card in protective sleeve",
        "Craft supplies around"
        ]

    styles = [
        "photorealistic", 
        "professional photography", 
        "premium product shot", 
        "museum display lighting"
        ]

    environmental = [
        "casino atmosphere", 
        "green felt table", 
        "wooden table", 
        "glass table",
        "marble countertop",
        "vintage shop display",
        "antique store shelf", 
        "magic shop interior",
        "cozy living room",
        "dimly lit room",
        "collector's desk"
        ]
    prompt = random.choice(base_prompts) + ", " + random.choice(styles) + ", " + random.choice(environmental)
    return prompt


def image_to_int_array(image_path, format="PNG", invert=False) -> list:
    """Current Workers AI REST API consumes an array of unsigned 8 bit integers"""
    image = Image.open(image_path)
    if invert:
        image = ImageOps.invert(image)

    bytes = io.BytesIO()
    image.save(bytes, format=format)
    return list(bytes.getvalue())


def cloudflare_inpainting(image_path: Union[str, Path], 
                         mask_path: Union[str, Path], 
                         prompt: str,
                         account_id: str,
                         api_token: str) -> bytes:
    """
    Consumes Cloudflare Workers AI API for Stable Diffusion Inpainting
    
    Args:
        image_path: Path to the reference image
        mask_path: Path to the mask image
        prompt: Descriptive text to generate content
        account_id: Cloudflare account ID
        api_token: Cloudflare API token
    
    Returns:
        bytes: Generated image in bytes format
    
    Raises:
        requests.RequestException: If there's an HTTP request error
        FileNotFoundError: If image files are not found
    """
    
    # API URL
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/runwayml/stable-diffusion-v1-5-inpainting"


    # Headers
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    try:
        # Prepare payload
        payload = {
            "image": image_to_int_array(image_path),
            "mask": image_to_int_array(mask_path, invert=True),  # Invert mask for inpainting
            "prompt": prompt,
            "height": 640,  # Optional: height of the output image
            "width": 480,   # Optional: width of the output image
            #"num_images": 1,  # Optional: number of images to generate
            "seed": 12345,    # Optional: random seed for reproducibility
            "num_steps": 20,  # Optional: number of diffusion steps
            "strength": 1,    # Optional: transformation strength (0-1)
            "guidance": 7.5   # Optional: guidance scale
        }
        
        # Make request
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Response contains generated image in base64
        #print(f"Response Code {response.status_code}")
        if response.status_code != 200:
            raise requests.RequestException(f"Error: {response.status_code} - {response.text}")

        return response.content
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find file: {e}")
    except requests.RequestException as e:
        raise requests.RequestException(f"HTTP request error: {e}")
    except KeyError as e:
        raise KeyError(f"Unexpected API response: {e}")




def save_generated_image(image_bytes: bytes, output_path: Union[str, Path]) -> None:
    """
    Saves the generated image to a file
    
    Args:
        image_bytes: Bytes of the generated image
        output_path: Path where to save the image
    """
    with open(output_path, "wb") as f:
        f.write(image_bytes)
    #print(f"Image saved at: {output_path}")




def main():
    # Cloudflare API key and image paths
    # https://dash.cloudflare.com/81b5d4ffd895b759a0f285ee954d8777/api-tokens
    # Verificar si existe
    if 'ACCOUNT_ID' in os.environ:
        ACCOUNT_ID = os.environ['ACCOUNT_ID']
    else:
        print("ACCOUNT_ID not found")
        exit(1)

    if 'API_TOKEN' in os.environ:
        API_TOKEN = os.environ['API_TOKEN']
    else:
        print("API_TOKEN not found")
        exit(1)

    input_image = os.path.realpath("../dataset/train/images/full_art_00b1139d-e87c-415d-be20-d4d31480ebdc_001.jpg")
    fg_mask = os.path.realpath("../dataset/train/masks/full_art_00b1139d-e87c-415d-be20-d4d31480ebdc_001.png")

    prompt = gen_prompt()
    print(f"Generated prompt: {prompt}")

    # Generate image using Cloudflare Workers AI
    result = cloudflare_inpainting(
        image_path=input_image,
        mask_path=fg_mask, 
        prompt=prompt,
        account_id=ACCOUNT_ID,
        api_token=API_TOKEN
    )
    
    # Save result
    save_generated_image(result, "generated_image.png")

if __name__ == "__main__":
    main()
