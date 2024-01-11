import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import csv

# Spotify API credentials
client_id = 'ab5b0b3e6a324e36a0d6d6d93f886852'
client_secret = '2dd80154c3a7443486ab3fab4c13383d'

# Set up Spotify API authentication
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Get the 500 most streamed songs of all time
playlist_id = '0JiVp7Z0pYKI8diUV6HJyQ'  # https://open.spotify.com/playlist/0JiVp7Z0pYKI8diUV6HJyQ?si=bfc4f6d0a7874ded
results = sp.playlist_tracks(playlist_id)

# Create and write to CSV file
with open('datasets/top_songs.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Title', 'Artist', 'Valence', 'Arousal']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for track in results['items']:
        title = track['track']['name']
        artist = track['track']['artists'][0]['name']
        # Get track ID
        track_id = track['track']['id']
        
        # Fetch audio features
        audio_features = sp.audio_features([track_id])[0]

        # Extract Valence and Arousal values
        valence = audio_features['valence']
        arousal = audio_features['energy']
        # Write the data to the CSV file
        writer.writerow({'Title': title, 'Artist': artist, 'Valence': valence, 'Arousal': arousal})
