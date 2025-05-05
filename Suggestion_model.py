import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google.colab import drive
import tensorflow as tf
import os

# Check for GPU availability
if not tf.config.list_physical_devices('GPU'):
    raise SystemError("GPU device not found. Please enable GPU in Colab settings.")
else:
    print("Using GPU for training.")

# Mount Google Drive
drive.mount('/content/drive')

# Step 1: Load and preprocess dataset
def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(subset=['category', 'objects', 'distance', 'caption'], inplace=True)

    # Combine relevant features to form input text
    df['input_text'] = df['category'] + ", " + df['objects'] + ", " + df['distance']

    return df['input_text'], df['caption']

# Step 2: Tokenize input and output texts
def tokenize_data(input_texts, captions, max_vocab=5000, max_seq_len=50):
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(input_texts + captions)

    input_sequences = tokenizer.texts_to_sequences(input_texts)
    caption_sequences = tokenizer.texts_to_sequences(captions)

    input_padded = pad_sequences(input_sequences, maxlen=max_seq_len, padding='post', truncating='post')
    caption_padded = pad_sequences(caption_sequences, maxlen=max_seq_len, padding='post', truncating='post')

    return input_padded, caption_padded, tokenizer

# Step 3: Build model
def build_model(vocab_size, embedding_dim=64, lstm_units=32, max_seq_len=50):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_seq_len),
        LSTM(lstm_units, return_sequences=False),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Train and save model
def train_and_save_model(input_padded, caption_padded, tokenizer, save_path):
    vocab_size = len(tokenizer.word_index) + 1
    max_seq_len = input_padded.shape[1]

    model = build_model(vocab_size, max_seq_len=max_seq_len)
    model.fit(input_padded, np.expand_dims(caption_padded[:, 0], axis=-1), epochs=100, batch_size=4, validation_split=0.2)
    model.save(save_path)

    return model

# Step 5: Generate new captions for multiple rows
def generate_combined_caption(model, tokenizer, input_texts, max_seq_len=50, lines_per_chunk=2):
    input_sequences = tokenizer.texts_to_sequences(input_texts)
    input_padded = pad_sequences(input_sequences, maxlen=max_seq_len, padding='post', truncating='post')

    predictions = model.predict(input_padded, verbose=0)
    predicted_indices = np.argmax(predictions, axis=-1)

    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    captions = [reverse_word_index.get(idx, "<Unknown>") for idx in predicted_indices]

    # Combine captions into chunks of lines_per_chunk lines
    combined_caption = "\n".join([" ".join(captions[i:i + lines_per_chunk]) for i in range(0, len(captions), lines_per_chunk)])
    return combined_caption

# Main script
if __name__ == "__main__":
    # Set paths for dataset and model
    dataset_path = "/content/drive/My Drive/datasets.csv"  # Replace with your dataset path in Google Drive
    model_save_path = "/content/drive/My Drive/caption_generator_model.h5"  # Replace with your model save path

    # Load and preprocess data
    input_texts, captions = preprocess_data(dataset_path)

    # Tokenize data
    input_padded, caption_padded, tokenizer = tokenize_data(input_texts, captions)

    # Train and save model
    model = train_and_save_model(input_padded, caption_padded, tokenizer, model_save_path)

    # Test input-output for multiple rows
    test_inputs = [
        "Attending a meeting or class (POV at a desk or table), 3 person, 0.64m, 0.64m, 2.39m",
        "Attending a meeting or class (POV at a desk or table), 2 person, 1 laptop, 0.58m, 0.62m, 2.27m",
        "Attending a meeting or class (POV at a desk or table), 4 person, 1.06m, 0.96m, 0.96m, 1.38m",
        "Brushing teeth (POV looking in the mirror),1 person,0.54m",
        "Charging a smartphone (POV with a charger and port visible),no objects,0m"
    ]
    combined_caption = generate_combined_caption(model, tokenizer, test_inputs)

    print("Input Rows:")
    for row in test_inputs:
        print(row)
    print("\nGenerated Combined Caption:")
    print(combined_caption)

