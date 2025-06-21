'''
pip3 install fastapi uvicorn
'''
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn


app = FastAPI(title="Demo",
              redoc_url=None,
              docs_url=None)



static_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "train","exported_models")
app.mount("/models", StaticFiles(directory=static_path, html=True), name="models")

static_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "demo")
app.mount("/", StaticFiles(directory=static_path, html=True), name="demo")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)