import numpy as np
import pandas
df = pandas.read_csv('datasets/top_songs.csv')
df = pandas.DataFrame(df)

def find_closest_song(df, v, a, n = 1):
    # Calculate Euclidean distance for each row
    df['distance'] = np.sqrt((df['Valence'] - v)**2 + (df['Arousal'] - a)**2)

    # Find the row with the minimum distance
    closest_rows = df.nsmallest(n, 'distance')

    # Get the title of the nearest row
    closest_titles = closest_rows['Title'].tolist()
    closest_artists = closest_rows['Artist'].tolist()
    closest_valences = closest_rows['Valence'].tolist()
    closest_arousals = closest_rows['Arousal'].tolist()

    # Drop the temporary 'distance' column
    df = df.drop(columns=['distance'])

    return closest_titles, closest_artists, closest_valences, closest_arousals
