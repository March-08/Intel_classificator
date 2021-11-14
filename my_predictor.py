import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt



def predict_with_model(model, imgpath):

    image = tf.io.read_file(imgpath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [150,150]) # (60,60,3)
    image = tf.expand_dims(image, axis=0) # (1,60,60,3)

    predictions = model.predict(image) # [0.005, 0.00003, 0.99, 0.00 ....]
    predictions = np.argmax(predictions) # 2

    return predictions



if __name__=="__main__":
    
    mapping = {
        'buildings' : 0,
        'forest' : 1,
        'glacier' : 2,
        'mountain' : 3,
        'sea' : 4,
        'street' : 5 
        }

    img_path = "D:\\DATASETS\\INTEL_IMAGE\\pred\\64.jpg"
    model = tf.keras.models.load_model('./Models')

    prediction = predict_with_model(model, img_path)

    print(f"prediction = {prediction}")
    print(list(mapping.keys())[list(mapping.values()).index(prediction)])
   
    # img = mpimg.imread(img_path)
    # imgplot = plt.imshow(img)
    # plt.show()

