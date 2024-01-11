# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset
df = pd.read_csv('datasets/anew.csv')

# Drop rows with missing values
df = df.dropna()

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Check for any missing values in the 'text' column
train_data = train_data.dropna(subset=['text'])
test_data = test_data.dropna(subset=['text'])

# Convert text data into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['text'])
X_test = vectorizer.transform(test_data['text'])

# Use linear regression for valence prediction
valence_model = LinearRegression()
valence_model.fit(X_train, train_data['valence'])
valence_predictions = valence_model.predict(X_test)

# Use linear regression for arousal prediction
arousal_model = LinearRegression()
arousal_model.fit(X_train, train_data['arousal'])
arousal_predictions = arousal_model.predict(X_test)

# Evaluate the models
valence_mse = mean_squared_error(test_data['valence'], valence_predictions)
arousal_mse = mean_squared_error(test_data['arousal'], arousal_predictions)

print(f'Mean Squared Error (Valence): {valence_mse}')
print(f'Mean Squared Error (Arousal): {arousal_mse}')

new_text = ["Type in Text here"]

# Convert the new text into numerical features using the same CountVectorizer
new_text_vectorized = vectorizer.transform(new_text)

# Use the trained models to predict valence and arousal
predicted_valence = valence_model.predict(new_text_vectorized)
predicted_arousal = arousal_model.predict(new_text_vectorized)

print(new_text)
print(f'Predicted Valence: {predicted_valence[0]}')
print(f'Predicted Arousal: {predicted_arousal[0]}')

