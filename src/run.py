# -*- coding: utf-8 -*-

# Internal imports
from app import app
from flask import Flask, app, request
import os
from os.path import join, dirname, realpath
from werkzeug.utils import secure_filename
from PIL import Image
from app.ai.predict import predict_with_model
import json
import tensorflow as tf


MODEL_PATH = "src/app/ai/models"


@app.route("/predict/", methods=["POST"])
def predict_image():
    UPLOADS_PATH = join(dirname(realpath(__file__)), "static\\img")
    if request.files["image"].filename != "":
        image = request.files["image"]
        filename = os.path.join(UPLOADS_PATH, secure_filename(image.filename))
        image.save(filename)
        # resize image (rewrite into utils)
        _, prediction = predict_with_model(model=model, imgpath=filename)
        return json.dumps(dict(prediction=prediction))


if __name__ == "__main__":
    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects=None, compile=True, options=None
    )
    app.run(debug=app.config.get("DEBUG", True))
