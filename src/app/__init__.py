# -*- coding: utf-8 -*-

# External imports
import os
import tensorflow as tf
from flask import Flask, jsonify
from werkzeug.utils import secure_filename

# Internal imports
from ai.predict import predict_with_model


try:
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', '/tmp/') 
    print('loading model..')
    model = tf.keras.models.load_model('./ai/models')
    print('model loaded successfully!')

    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

    def allowed_filename(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/predict', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return jsonify({'result': 'no file provided.'}), 400
        file = request.files['file']
        if file.filename == '' or not allowed_filename(file.filename):
            return jsonify({'result': 'invalid filename.'}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], name=filename)
        file.save(filepath)
        prediction, mapped_result = predict_with_model(model, filepath)
        os.remove(filepath)
        return jsonify({'result': {'prediction': prediction, 'mapped_result': mapped_result}}), 200

except Exception as err:
    print('an error occurred while starting app: %s' % (str(err), ))
