# College Enquiry Chatbot
Overview

This is an AI-powered chatbot designed to answer queries related to BGS Institute of Technology (BGSIT). The chatbot processes user inputs and responds with relevant information about admissions, courses, placements, facilities, and more. It is built using Python, TensorFlow, NLTK, and NumPy with Natural Language Processing (NLP) techniques.

Features

Provides information about the college (BGSIT) and its courses

Answers common queries related to admissions, placements, and facilities

Uses a trained deep learning model for intent recognition

Allows customization by modifying intents.json

Can be adapted for other institutions with relevant data changes

Installation

Prerequisites

Make sure you have Python installed. Then install the required dependencies:

pip install -r requirements.txt

How to Use

Step 1: Train the Model

Before running the chatbot, you need to train the model:

python training.py

This will:

Process the intents from intents.json

Train a neural network model using TensorFlow and NLTK

Save the trained model as chatbotmodel.h5

Store processed data in words.pkl and classes.pkl

Step 2: Run the Chatbot

Once the model is trained, run the chatbot using:

python main.py

You can then ask questions, and the chatbot will respond accordingly.

Customization

Editing Responses

To modify the chatbot’s responses, edit intents.json. You can add new intents or update existing ones:

{
  "tag": "new_intent",
  "patterns": ["User question here"],
  "responses": ["Bot response here"]
}

After making changes, retrain the model by running training.py again.

Adapting for Other Institutions

This chatbot is designed for BGSIT, but it can be customized for any institution by modifying:

intents.json to include relevant details

Training the model again to reflect the new data

Files in the Repository

intents.json - Contains predefined questions and responses

training.py - Processes data and trains the chatbot model

trainingData.py - Alternative script for training

main.py - Runs the chatbot and handles user interactions

requirements.txt - Dependencies required for the project

chatbotmodel.h5 - Trained model file (generated after training)

words.pkl & classes.pkl - Processed data files (generated after training)

Future Enhancements

Improve accuracy with advanced NLP techniques

Add support for voice-based interactions

Integrate with a web-based interface
