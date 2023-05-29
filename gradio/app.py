import gradio as gr
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tqdm import tqdm
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
def clean_text(raw_text):
    # Remove unnecessary symbols and numbers
    cleaned_text = re.sub('[^a-zA-Z]', ' ', raw_text)
    
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    
    # Tokenize the text
    words = cleaned_text.split()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Perform stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the cleaned words back into a single string
    cleaned_text = ' '.join(words)
    
    return cleaned_text
# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained("troll_model")

def predict_text(input_text):
    # Tokenize and encode the input text
    input_ids = tokenizer.encode_plus(
        clean_text(input_text),
        add_special_tokens=True,
        max_length=128,
        padding='longest',
        truncation=True,
        return_tensors='tf'
    )['input_ids']

    # Make prediction
    predictions = model.predict(input_ids)[0]

    # Get predicted label and confidence scores
    predicted_label = tf.argmax(predictions, axis=1).numpy()[0]
    confidence_scores = tf.nn.softmax(predictions, axis=1).numpy()[0]

    # Interpret the predicted label
    if predicted_label == 0:
        output_text = f"Not troll, Troll level: {confidence_scores[1]}"
    else:
        output_text = f"Troll, Troll level: {confidence_scores[1]}"

    return output_text

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_text,
    inputs="text",
    outputs="text",
    title="Text Classification",
    description="Enter a text and the model will predict its class.",
    theme="default"
)

# Launch the interface
iface.launch()
