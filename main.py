import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import string
import os
from constant import DATA_PATH, DATA_NOT_FOUND, CHARACTERS_COUNT, VOCAB_COUNT, INPUT, OUTPUT, SEQUENCES_COUNT

# TASK 1 : Dataset Loading and Preprocessing

dataset_path = DATA_PATH
# check dataset file are proper found or not
if not os.path.exists(dataset_path):
    raise Exception(DATA_NOT_FOUND)

# if dataset file found then is open and read
with open(dataset_path, 'r', encoding='utf-8') as file:
    text = file.read()

print(CHARACTERS_COUNT.format(len(text)))


#  Dataset Preprocessing and Convert the text to lowercase and remove punctuation
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


text = preprocess_text(text)

# Tokenize the text into sequences of words or characters
chars = sorted(list(set(text)))
char_index = {char: index for index, char in enumerate(chars)}
index_char = {index: char for char, index in char_index.items()}

vocab_size = len(chars)
print(VOCAB_COUNT.format(vocab_size))

""" Prepare input-output pairs where the input is a sequence of tokens, and the output is
the next token in the sequence."""

encoded_text = np.array([char_index[c] for c in text])
seq_length = 100
step = 5

x = []
y = []

for i in range(0, len(encoded_text) - seq_length, step):
    x.append(encoded_text[i:i+seq_length])
    y.append(encoded_text[i+seq_length])

X = np.array(x)
Y = np.array(y)

print(INPUT.format(X.shape))
print(OUTPUT.format(Y.shape))
print(SEQUENCES_COUNT.format(len(X)))

# Task 2 : Model Design:
vocab_size = len(chars)
seq_length = 100
embedding_dim = 64
lstm_units = 128

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    LSTM(lstm_units),
    Dense(vocab_size, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.build(input_shape=(None, seq_length))
model.summary()

# 3 Model Training:
# Split the dataset into training and validation sets
split_idx = int(0.9 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = Y[:split_idx], Y[split_idx:]

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Model checkpoint callback
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=128,
    epochs=20,
    callbacks=[early_stopping, checkpoint]
)


# 4 Text Generation
def generate_text(model, start_string, gen_length=500):
    """
    Generate text using a trained character-level language model.

    Args:
        model : Trained character-level text generation mode
        start_string: text to start generation
        gen_length : Number of characters to generate
    Returns:
        str: Generated text starting from the start_string followed by gen_length characters
    """

    # Convert each character in the seed string to its corresponding index
    input_eval = [char_index[s] for s in start_string.lower() if s in char_index]
    input_eval = tf.expand_dims(input_eval, 0)

    generated_text = []

    for _ in range(gen_length):
        # Predict the next character
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predicted_id = tf.random.categorical(predictions[None, :], num_samples=1)[-1, 0].numpy()

        next_char = index_char[predicted_id]
        generated_text.append(next_char)

        input_eval = tf.concat([input_eval[:1:], [[predicted_id]]], axis=-1)

    # Combine the original seed text with the generated characters
    return start_string + ''.join(generated_text)


# Generate text starting with "To be, or not to be"
print(generate_text(model, start_string="To be, or not to be"))