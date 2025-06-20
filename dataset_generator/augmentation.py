'''

inpaint outpaint
'''
import os



def main():
    API_KEY="SG_590c39cf55bde8d0"
    input_image = os.path.realpath("../dataset/train/images/full_art_00b1139d-e87c-415d-be20-d4d31480ebdc_001.jpg")
    fg_mask = os.path.realpath("../dataset/train/masks/full_art_00b1139d-e87c-415d-be20-d4d31480ebdc_001.png")

    prompt = "a hand holding a card, random background, high quality, detailed, realistic"

if __name__ == "__main__":
    main()
