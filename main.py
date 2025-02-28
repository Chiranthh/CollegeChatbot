import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents, words, classes and the trained model
try:
    intents = json.loads(open('intents.json', 'r', encoding='utf-8').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbotmodel.h5')
except Exception as e:
    print("Error loading files:", str(e))
    print("Please make sure all required files exist and run training.py first.")
    exit(1)

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the sentence"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Convert sentence to bag of words"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predict the class of the sentence"""
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """Get a random response from the intent"""
    if not intents_list:
        return "I apologize, but I'm not sure I understood your question. Could you please rephrase it or ask about something specific like our courses, facilities, or admission process?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    # Add spell check suggestion if confidence is low
    probability = float(intents_list[0]['probability'])
    if probability < 0.5:
        return "I'm not entirely sure what you're asking. Did you mean to ask about our courses, facilities, or admission process?"
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "I apologize, but I'm not sure how to respond to that. Could you please ask about something specific like our courses, facilities, or admission process?"
    
    return result

def main():
    """Main function to run the chatbot"""
    print("|============= Welcome to College Enquiry Chatbot System! =============|")
    print("|============================== Feel Free ============================|")
    print("|================================== To ===============================|")
    print("|=============== Ask your any query about our college ================|")
    
    while True:
        try:
            message = input("| You: ").strip()
            if not message:
                continue
                
            if message.lower() in ['bye', 'goodbye', 'exit', 'quit']:
                ints = predict_class(message)
                res = get_response(ints, intents)
                print("| Bot:", res)
                print("|===================== The Program End here! =====================|")
                break
            
            ints = predict_class(message)
            res = get_response(ints, intents)
            print("| Bot:", res)
            
        except KeyboardInterrupt:
            print("\n|===================== Program terminated! =====================|")
            break
        except Exception as e:
            print("| Bot: Sorry, I encountered an error. Please try again.")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()