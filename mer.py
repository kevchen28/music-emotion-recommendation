from knn import predict_valence_arousal
from search_song import find_closest_song
import pandas

df = pandas.read_csv('datasets/top_songs.csv')
df = pandas.DataFrame(df)

num_recommendations = int(input("How many songs would you like recommended? "))
user_input = input("Enter a string: ").lower()

while (user_input):
    predicted_valence, predicted_arousal = predict_valence_arousal(user_input)

    if predicted_valence is not None and predicted_arousal is not None:
        print(f"Predicted Valence for '{user_input}': {predicted_valence}")
        print(f"Predicted Arousal for '{user_input}': {predicted_arousal}")

        closest_titles, closest_artists, closest_valences, closest_arousals = find_closest_song(df, predicted_valence, predicted_arousal, num_recommendations)

        for i in range(len(closest_titles)):
            print("Artist: {}, Song Title: {}, Valence: {}, Arousal: {}".format(closest_titles[i], closest_artists[i], closest_valences[i], closest_arousals[i]))

    else:
        print(f"Word '{user_input}' not found in the word2vec model.")

    user_input = input("Enter a string (type x to exit): ").lower()

    if user_input == 'x':
        exit()
