import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import tkinter as tk
from tkinter import simpledialog, messagebox

# Load dataset
df = pd.read_csv("movies.csv")

# Handling NaNs and infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Define numerical and categorical columns
numerical_cols = ['Year', 'Runtime', 'No.of.Ratings']
categorical_cols = ['Certificate', 'Overview', 'Movie']

# Fill missing values in numerical columns with mean
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Fill missing values in categorical columns
df['Certificate'].fillna('Unknown', inplace=True)
df['Overview'].fillna('', inplace=True)

# Encode 'Certificate' feature
certificate_encoder = OneHotEncoder()
certificate_encoded = certificate_encoder.fit_transform(df[['Certificate']]).toarray()

# Combine numerical and encoded features
X_numerical = df[numerical_cols].values
X_combined = np.concatenate([X_numerical, certificate_encoded], axis=1)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_combined)

# Simulate interaction data (user, movie, timestamp)
interactions = pd.DataFrame({
    'user_id': np.random.randint(1, 100, 1000),
    'movie_id': np.random.choice(df.index, 1000),
    'timestamp': pd.date_range(start='2020-01-01', periods=1000, freq='H')
})

# Merge interaction data with movie data
interactions = interactions.merge(df, left_on='movie_id', right_index=True)
interactions = interactions.merge(pd.DataFrame(X_scaled, columns=[f'feature_{i}' for i in range(X_scaled.shape[1])]), left_on='movie_id', right_index=True)
interactions = interactions.sort_values(by=['user_id', 'timestamp'])

# Function to create sequences
def create_sequences(interactions, max_sequence_length):
    sequences = []
    for user_id, group in interactions.groupby('user_id'):
        user_sequences = group.iloc[:, -X_scaled.shape[1]:].values
        for i in range(len(user_sequences) - 1):
            sequences.append(user_sequences[max(0, i-max_sequence_length):i+1])
    return sequences

max_sequence_length = 20  # Set desired sequence length
sequences = create_sequences(interactions, max_sequence_length)

X = np.zeros((len(sequences), max_sequence_length, X_scaled.shape[1]), dtype=np.float32)
y = np.zeros((len(sequences), max_sequence_length), dtype=np.int32)

for i, seq in enumerate(sequences):
    seq_length = min(len(seq), max_sequence_length)
    for t in range(seq_length):
        X[i, t, :] = seq[t]
        if t < seq_length - 1:
            y[i, t] = seq[t+1, 1]
    if seq_length < max_sequence_length:
        X[i, seq_length:, :] = seq[-1]

X = X[:, :max_sequence_length, :]
y = y[:, :max_sequence_length]

# Convert y to one-hot encoding
num_classes = len(df)
y_train_onehot = to_categorical(y, num_classes=num_classes).astype(np.float32)

# Define and compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

model = Sequential([
    GRU(units=128, input_shape=(max_sequence_length, X_scaled.shape[1]), return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=l2(0.01)),
    Dropout(0.5),  # Increase dropout for more regularization
    GRU(units=128, return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(units=num_classes, activation='softmax')
])

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y_train_onehot, test_size=0.2, random_state=42)

history = model.fit(X_train, y_train, epochs=20, batch_size=200, validation_data=(X_test, y_test), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

# Function to extract features by movie name
def extract_features_by_name(movie_name):
    movie_row = df[df['Movie'] == movie_name]
    if movie_row.empty:
        raise ValueError(f"Movie '{movie_name}' not found in the dataset.")
    entered_movie_features = movie_row[numerical_cols].values
    certificate_encoded = certificate_encoder.transform(movie_row[['Certificate']]).toarray()
    entered_movie_features_combined = np.concatenate([entered_movie_features, certificate_encoded], axis=1)
    entered_movie_features_scaled = scaler.transform(entered_movie_features_combined)
    return entered_movie_features_scaled.reshape(1, 1, -1)

# Function to recommend movies
def recommend_movies(user_id, entered_movie_name, top_n=10):
    
    user_sequence = interactions[interactions['user_id'] == user_id].iloc[:, -X_scaled.shape[1]:].values
    user_sequence_padded = np.zeros((1, max_sequence_length, X_scaled.shape[1]))
    user_sequence_length = min(max_sequence_length - 1, len(user_sequence))
    user_sequence_padded[0, 1:user_sequence_length+1, :] = user_sequence[-user_sequence_length:]
    entered_movie_features = extract_features_by_name(entered_movie_name)
    user_sequence_with_movie = np.concatenate([user_sequence_padded, entered_movie_features], axis=1)
    user_sequence_with_movie = user_sequence_with_movie[:, :max_sequence_length, :]
    user_preferences = model.predict(user_sequence_with_movie)[0][-1]
    top_indices = np.argsort(user_preferences)[::-1][:top_n]
    recommended_movie_names = df.iloc[top_indices]['Movie'].values
    return recommended_movie_names

# GUI for input and output
def get_movie_name():
    movie_name = simpledialog.askstring("Movie Recommendation", "Enter the movie name:")
    user_id = random.randint(1,1000)
    if movie_name:
        recommended_movies = recommend_movies(user_id, movie_name, top_n=10)
        result_text = "\n".join(recommended_movies)
        messagebox.showinfo("Recommended Movies", f"Recommended Movies:\n{result_text}")
        get_movie_name()

# Define user_id
user_id = random.randint(1,1000)

# Run GUI
root = tk.Tk()
root.withdraw()  # Hide the root window
get_movie_name()
root.mainloop()