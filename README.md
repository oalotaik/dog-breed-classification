# Installations
In order to run the codes in this project, the following libraries must be installed:
1. Keras
2. Numpy
3. Tensorflow
4. Flask
5. OpenCV


# Motivation
This project was done to complete the requirements for Udacity's Data Scientist Nanodegree. The task is to build a Convolutional Neural Network (CNN) classifier that, given a dog or person image, tells which breed the dog is or the breed that most resembles the person. If the given image is neither a dog nor a person, the classifier should detect that.

# Files
The project is divided into 6 folders: one for the bottleneck features of the CNN (bottleneck_features); another one is for the human detector (haarcascades); another one is for the trained model (model); then (static), (templates), and (uploads) are for the web app.

### Files in bottleneck_features
1. DogResnet50Data.npz: numpy file containing the bottleneck features for the Resnet50 model

### Files in haarcascades
1. haarcascade_frontalface_alt.xml: XML file containing the model for human detection

### Files in the model folder
1. weights for the trained model

### Files in static
1. CSS file for the front-end of web app

### Files in templates
1. two HTML files for the web app, index.html and template.html

### Files in uploads
1. a default picture for the web app
2. future uploaded pictures from users of the web app

### Remaining files
1. app.py: flask web app
2. extract_bottleneck_features.py: include reference functions used in app.py
3. names_dogs.py: a list containing the breed labels
4. screenshot.JPG: screenshot of the web app
5. dog_app.ipynb: jupyter notebook 

# Instructions for running the app:
The web app is not hosted. So, to run it, follow the instructions below:
1. Download all files and folders in the same order they are organized here
2. Run the following command in the app's directory to run your web app: `python app.py`
3. The app will run on a local server; get the address given by the app, e.g. http://127.0.0.1:5000
4. In the browser, go to the given address, e.g. http://127.0.0.1:5000

# Results
The final output of the project is an interactive web app that takes an image from the user as an input and then classifies it.

# Screenshot
![Web app](https://github.com/oalotaik/dog-breed-classification/blob/master/screenshot.JPG)

# Acknowledgement
Thanks to Udacity for providing guidance to complete the project
