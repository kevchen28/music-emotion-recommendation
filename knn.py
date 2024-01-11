import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from gensim.models import KeyedVectors
from gensim import downloader as api

# Load ANEW dataset
anew_df = pd.read_csv("datasets/anew.csv")

# Load Word2Vec model (replace 'path/to/word2vec_model' with your Word2Vec model file)
word2vec_model = api.load("word2vec-google-news-300")  # Load in model

k = 10 #How many nearest neighbors

# Function to get word vector using word2vec model
def get_word_vector(word):
    try:
        return word2vec_model[word]
    except KeyError:
        return None

# Apply word2vec to get vectors for words in ANEW dataset
anew_df['WordVector'] = anew_df['text'].apply(get_word_vector)

# Drop rows with missing word vectors
anew_df = anew_df.dropna(subset=['WordVector'])

# Split the dataset into features (X) and target values (y)
X = np.vstack(anew_df['WordVector'].to_numpy())
y_valence = anew_df['valence'].to_numpy()
y_arousal = anew_df['arousal'].to_numpy()

# Split the dataset into training and testing sets
X_train, X_test, y_valence_train, y_valence_test, y_arousal_train, y_arousal_test = train_test_split(
    X, y_valence, y_arousal, test_size=0.2, random_state=42
)

# Initialize KNN regressor for valence
knn_valence = KNeighborsRegressor(n_neighbors=k, weights="distance")
knn_valence.fit(X_train, y_valence_train)

# Initialize KNN regressor for arousal
knn_arousal = KNeighborsRegressor(n_neighbors=k, weights="distance")
knn_arousal.fit(X_train, y_arousal_train)

# Evaluate the model on the test set and calculate MSE
y_valence_pred = knn_valence.predict(X_test)
y_arousal_pred = knn_arousal.predict(X_test)

mse_valence = mean_squared_error(y_valence_test, y_valence_pred)
mse_arousal = mean_squared_error(y_arousal_test, y_arousal_pred)

print(f'Mean Squared Error (Valence): {mse_valence}')
print(f'Mean Squared Error (Arousal): {mse_arousal}')

# Predict valence and arousal for unseen words using word2vec similarity
def predict_valence_arousal(sentence):
    words = sentence.split()
    valence_values = []
    arousal_values = []

    for word in words:
        word_vector = get_word_vector(word)
        if word_vector is not None:
            # Use KNN to predict valence and arousal
            valence = knn_valence.predict([word_vector])[0]
            arousal = knn_arousal.predict([word_vector])[0]
            valence_values.append(valence)
            arousal_values.append(arousal)

    if valence_values and arousal_values:
        # Calculate the mean valence and arousal values for the sentence
        mean_valence = np.mean(valence_values)
        mean_arousal = np.mean(arousal_values)
        return mean_valence, mean_arousal
    else:
        return None, None
