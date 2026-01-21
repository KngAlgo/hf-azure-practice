from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

dataset = load_dataset('imdb')
train_data = dataset['train']
test_data = dataset['test']

tokenized_train = tokenizer(train_data['text'], truncation=True, padding=True, return_tensors="tf")
tokenized_test = tokenizer(test_data['text'], truncation=True, padding=True, return_tensors="tf")

outputs = model(**tokenized_test)
logits = outputs.logits
pred = tf.argmax(logits, axis=-1)