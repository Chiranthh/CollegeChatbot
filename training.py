import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import os

# Create a directory for model files if it doesn't exist
os.makedirs('model', exist_ok=True)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def load_and_process_intents():
    """Load and process the intents file"""
    try:
        with open('intents.json', 'r', encoding='utf-8') as file:
            intents = json.load(file)
    except Exception as e:
        print(f"Error loading intents file: {str(e)}")
        exit(1)
    
    return intents

def prepare_training_data(intents):
    """Prepare training data from intents"""
    lemmatizer = WordNetLemmatizer()
    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '.', ',']

    # Process patterns and intents
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lemmatize and clean words
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    return words, classes, documents

def create_training_data(words, classes, documents):
    """Create training data for the model"""
    lemmatizer = WordNetLemmatizer()
    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)
    
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    
    return np.array(train_x), np.array(train_y)

def create_model(train_x, train_y):
    """Create and compile the model"""
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

def main():
    """Main function to train the model"""
    print("Loading and processing intents...")
    intents = load_and_process_intents()
    
    print("Preparing training data...")
    words, classes, documents = prepare_training_data(intents)
    
    print("Creating training datasets...")
    train_x, train_y = create_training_data(words, classes, documents)
    
    print("Creating and training the model...")
    model = create_model(train_x, train_y)
    hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
    
    print("Saving the trained model and data...")
    model.save('chatbotmodel.h5')
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))
    
    print("Training completed successfully!")
    print(f"Final accuracy: {hist.history['accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main() 