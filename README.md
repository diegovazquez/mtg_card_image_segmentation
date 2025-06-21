# Magic The Gathering - Cards Image Segmentation

This project provides tools for generating Magic The Gathering Cards image segmentation machine learning model.

# Glossary

- **Magic The Gatering** Good card game for the cool people!
- **Machine Learning** A field of artificial intelligence
- **Python** Programing Language
- **Blender** 3D modeling and rendering Open Source software
- **BlenderProc2** A procedural Blender pipeline (tool) for photorealistic rendering
- **HDRI** High Dynamic Range Image, for creating realistic environments
- **Skryfall** MTG card database
- **Polyhaven** Website for 3d assets
- **API** An interface for talking with websites/apps
- **ONNX** Open Neural Network Exchange 
- **GAN** Generative Adversarial Network (IA image generators for example)

# The Plan

- [x] Create MTG Cards Syntetic Dataset Generator based on 3d redering, and generate the first version of this dataset. 

- [x] Create the machine learning training scrips and train the model

- [X] Transform to ONNX

- [X] Crate a web example for runing the model in web

Others: 

- [ ] Expand the dataset with a generative network like flux or Stable diffusion

# Hardware/Software Requirements

- 16 gb of ram or more (recomended)
- 10 GB of hard disk space (or more)
- A cuda 12 capable GPU (Nvidia) (others may be can be used, not tested)
- A working pytorch enviroment (google it)
- Ubuntu 24.04 is recomended

## Step 1 - MTG Cards Syntetic Dataset Generator (`dataset_generator/`)

**NOTE:** You cand download pregenerated dataset from this [link](https://huggingface.co/datasets/dhvazquez/mtg_synthetic_cards_semantic_segmentation/).

### 1. Reference Dataset Downloader (`01_download_references.py`)
Downloads MTG card reference images from Scryfall API.

### 2. Reference Dataset Downloader (`02_dowload_hdrs.py`)
Downloads realistic environments and lightings (HDRI) from Polyhaven API.

### 3. Synthetic Image Generator (`03_generate_synthetic_dataset.py`)
BlenderProc2-based synthetic image generator that creates realistic MTG card images with segmentation masks.

#### Features:
- **Photorealistic rendering** using HDRI environments
- **Precise card geometry** with rounded corners
- **Random transformations** (0-15° rotation on all axes)
- **Variable positioning** (40-70% image coverage)
- **Automatic segmentation masks** generation

#### Examples:
![example 1 mask](docs/example_1.png)
![example 1 mask](docs/example_1_mask.png)  
![example 2 mask](docs/example_2.png) 
![example 2 mask](docs/example_2_mask.png)

#### Quick Start:

```bash
# Create and activate python virtual enviroment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip package manager 
python -m pip install --upgrade pip

# Install dependencies
python -m pip install -r requirements.txt
blenderproc pip install tqdm>=4.66.0

# Download reference images
cd dataset_generator
python 01_download_references.py

# Download enviroments
python 02_dowload_hdrs.py

# Generate synthetic images
cd ..
blenderproc run dataset_generator/03_generate_synthetic_dataset.py

# For large datasets, use synthetic_dataset_anti_leak.py (restar the 03_generate_synthetic_dataset script every X minutes)
python synthetic_dataset_anti_leak.py -t 5

# blenderproc recomend not redering too many imgs in a single bender instance. I read this after the implementation...
# So... i make a dirty hack (it works!) 
```

Note: In Windows, if blenderproc is not in your path, you can try (examples):

    python -m blenderproc pip install tqdm>=4.66.0
    python -m blenderproc run dataset_generator/03_generate_synthetic_dataset.py

## Step 2 - Train  (`train/`)

Please read this documentation [link](train/README.md).


## Step 3 - Demo  (`demo/`)

#### Quick Start:

```bash
pip install fastapi uvicorn
python demo.py
INFO:     Started server process [5926]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit)
```

Go to http://127.0.0.1:5000/ in you browser.

Note: You need the exported model in train/exported_models/card_segmentation.onnx

# Developer Notes

This tools where created using: 

    VSCode
    Cline - AI Dev Assistant / VSCode extension
    Copilot - IA Dev Asistant (provider) / VSCode extension
    Claude 4 - IA Model
    Gemini 2.5 pro - IA Model
    Python / Python Debugger / Pylance - VSCode extensions

See [Synthetic Image Generator](dataset_generator/generate_synthetic.md) documentation.  
Original repo [link](https://github.com/diegovazquez/mtg_card_image_segmentation).  

Sorry for my english, im still learning.
