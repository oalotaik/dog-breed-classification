import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, load_model
import tensorflow as tf
#from keras import backend as K
import numpy as np
import argparse
import imutils
import cv2
import time
import uuid
import base64
from extract_bottleneck_features import *
from PIL import Image
from glob import glob




def path_to_tensor(img_path):
    '''Convert the image stored at img_path into 4D tensor'''
    # loads RGB image as PIL.Image.Image type
    img = load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    x = np.expand_dims(x, axis=0)
    return x #np.expand_dims(x, axis=0)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

def face_detector(img_path):
    '''Returns True if face is detected in image stored at img_path, False if no face detected'''
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

from keras.applications.resnet50 import ResNet50

global model
global graph
graph = tf.get_default_graph()

# define ResNet50 model
with graph.as_default():
    ResNet50_model_dog = ResNet50(weights='imagenet', input_shape=(224, 224, 3), include_top=False)

from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_dog(img_path):
    '''Returns prediction vector for image located at img_path'''
    img = preprocess_input(path_to_tensor(img_path))
    with graph.as_default():
        pred = np.argmax(ResNet50_model_dog.predict(img))

    return pred

def dog_detector(img_path):
    '''Returns "True" if a dog is detected in the image stored at img_path'''
    #with graph.as_default():
    prediction = ResNet50_predict_dog(img_path)
    return ((prediction <= 268) & (prediction >= 151))

# load list of dog_names
from names_dogs import names as dog_names

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])




with graph.as_default():
    # build pre-trained ResNet50 model
    bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
    Resnet50_model = Sequential()
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=bottleneck_features['train'].shape[1:]))
    Resnet50_model.add(Dense(512, activation='relu'))
    Resnet50_model.add(Dropout(0.2))
    Resnet50_model.add(Dense(256, activation='relu'))
    Resnet50_model.add(Dropout(0.1))
    Resnet50_model.add(Dense(133, activation='softmax'))
    # Compile the model.
    Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # load optimized model weights
    Resnet50_model.load_weights('model/weights.best.Resnet50.hdf5')


def Resnet50_predict_breed(img_path):
    '''Predict dog breed for the image stored at img_path with ResNet50 model'''
    # extract bottleneck features
    bottleneck_features = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    with graph.as_default():
        predicted_vector = Resnet50_model.predict(bottleneck_features)

    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def dog_breed_pred(path):
    '''Returns predicted dog breed if there is a dog or the resembling dog
    breed if a human is detected in image stored at path'''
    # Detect dog or human and run prediction
    if dog_detector(path):
        dog_breed = Resnet50_predict_breed(path)
        # formatting the breed name
        #dog_breed = dog_breed.split('.')
        #dog_breed = dog_breed[-1]
        result = 'This dog looks like a ' + dog_breed + '.'
    elif face_detector(path):
        resembling_dog_breed = Resnet50_predict_breed(path)
        # formatting the breed name
        #resembling_dog_breed = resembling_dog_breed.split('.')
        #resembling_dog_breed = resembling_dog_breed[-1]
        result = 'The most resembling dog breed of this person is ' + resembling_dog_breed + '.'
    else:
        result = 'There is no human or dog detected in this image.'
    return result

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='/uploads/what_is_it.jpg')
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            #with graph.as_default():
            result = dog_breed_pred(file_path)
            #label = result

            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=result, imagesource='/uploads/' + filename)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware

app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run()
