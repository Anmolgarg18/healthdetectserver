from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import cv2


import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import imageio as iio


origins = [
    "http://localhost:3000",
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post('/upload')
async def uploadFiles(file: bytes = File(...)):
    if file is not None:
        try:
            print('Something')
            model_path = "potatoes.h5"
            model = load_model(model_path)
            image=Image.open(io.BytesIO(file))
            image.save('temp.jpg')
            img = iio.imread('temp.jpg')
            print('iio',img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            print('color change',img)
            img = Image.open('temp.jpg')
            img = img.resize((256,256))
            print('Resize',img)
            img_array = np.array(img)
            print('np array',img)
            input_data = np.expand_dims(img_array, axis=0)
            print('input data',img)
            input_data = input_data / 255.0  # Normalize the input if necessary
            output = model.predict(input_data)
            print(output)
            classes = ['Early_blight','Late_Blight','Healthy']
            msg=classes[output.argmax()]
            print(classes[output.argmax()])
            return {'msg':msg,'success':True}
        except:
            return {'msg':'Some Error','success':False}
    else:
        return {'msg':'Opps Error','success':False}