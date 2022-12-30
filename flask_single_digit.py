from flask import Flask, render_template
import flask
from flask import request
import torch
import torchvision
from Model import Model
import cv2
from PIL import Image
import io
import numpy as np
from torchvision import transforms

app = Flask(__name__)

model = torch.load('single_digit.pth', map_location='cpu')
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

def run_inference(in_tensor):
    output = model(in_tensor)
    _, pred = torch.max(output, 1)
    prediction = pred.numpy()[0]
    return str(prediction)

@app.route('/predict', methods=['POST'])
def predict():
    data = {"success": False}
    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        # print(flask.request.files)
        if flask.request.files.get("image"):
        # Read the image in PIL format
            img = flask.request.files["image"].read()
            # img = Image.open(io.BytesIO(img))
        # print(type(img))
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        # Preprocess the image and prepare it for classification.
        # img = cv2.resize(img, (32, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(img)
            img = Image.fromarray(img)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor()
            ])

            image = transform(img)
            image = image.view(1, *image.size())

            data['prediction'] = run_inference(image)

        # Indicate that the request was a success.
            data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)

if __name__ == '__main__':
    app.run()