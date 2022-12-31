import torch
from torch.autograd import Variable
from tools import utils
from data_loader.data_loader import ResizeNormalize
import cv2
import models.crnn as crnn
import argparse
import flask
from config.config import Config
from flask import Flask, render_template
import numpy as np

config = Config()

# model_path = 'outputs/netCRNN_25_end.pth'

# net init
nclass = len(config.label_classes) + 1
# model = crnn.CRNN(config.img_height, config.nc, nclass, config.hidden_size, config.n_layers)

# load model
# model.load_state_dict(torch.load(model_path, map_location='cpu'))
model = torch.load('netCRNN.pth', map_location='cpu')
model.eval()

converter = utils.strLabelConverter(config.label_classes)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def run_inference(image):
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.LongTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))
    return sim_pred

@app.route('/predict', methods=['POST'])
def predict():
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            # Read the image in PIL format
            img = flask.request.files["image"].read()
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            # Preprocess the image and prepare it for classification.
            transformer = ResizeNormalize(config.img_height, config.img_width)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(img)
            image = transformer(img)
            image = image.view(1, *image.size())
            image = Variable(image)

            data['prediction'] = run_inference(image)

            # Indicate that the request was a success.
            data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)