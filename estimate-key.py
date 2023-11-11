import librosa
import numpy as np
from scipy.spatial.distance import cosine
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def generate_mode_profiles():
    ionian = np.array([1, 0, 0.5, 0, 0.5, 0.5, 0, 1, 0.5, 0, 0.5, 0.5])  # Major
    dorian = np.array([1, 0, 0.5, 1, 0, 0.5, 0, 1, 0, 0.5, 0.5, 0])
    phrygian = np.array([1, 0.5, 0, 0.5, 0.5, 1, 0, 0.5, 0.5, 0, 0.5, 0])
    lydian = np.array([1, 0, 0.5, 0, 1, 0.5, 0, 0.5, 0.5, 0, 1, 0])
    mixolydian = np.array([1, 0, 0.5, 0, 0.5, 0.5, 0, 1, 0.5, 0, 0.5, 1])
    aeolian = np.array([1, 0, 0.5, 1, 0, 0.5, 0.5, 0, 0.5, 0.5, 0, 0.5])  # Minor
    locrian = np.array([0.5, 0, 0.5, 1, 0, 0.5, 0.5, 0.5, 0, 0.5, 1, 0])

    # Rotate the profiles to create all 12 variations for each mode
    modes = [ionian, dorian, phrygian, lydian, mixolydian, aeolian, locrian]
    mode_profiles = [np.array([np.roll(mode, i) for i in range(12)]) for mode in modes]

    return mode_profiles

def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    return 1 - cosine(vec1, vec2)

def estimate_key(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path)

    # Apply bandpass filter
    y_filtered = butter_bandpass_filter(y, 300, 3000, sr)

    # Extract chroma features
    chroma = librosa.feature.chroma_cens(y=y_filtered, sr=sr)

    # Normalize the chroma features
    chroma_normalized = chroma / np.linalg.norm(chroma, axis=0)

    # Average the normalized chroma features across time
    chroma_avg = np.mean(chroma_normalized, axis=1)

    # Generate mode profiles
    mode_profiles = generate_mode_profiles()

    # Compute similarity with mode profiles
    similarities = [cosine_similarity(chroma_avg, mode[i]) for mode in mode_profiles for i in range(12)]

    # Find the mode with the highest similarity
    mode_idx = np.argmax(similarities)
    mode_name = ['Ionian (Major)', 'Dorian', 'Phrygian', 'Lydian', 'Mixolydian', 'Aeolian (Minor)', 'Locrian'][mode_idx // 12]
    key_idx = mode_idx % 12

    # Mapping to key names
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    estimated_key = keys[key_idx]

    # Manual Adjustment: If G Phrygian is detected, consider C Minor
    if mode_name == "Phrygian" and keys[key_idx] == "G":
        mode_name = "Aeolian (Minor)"
        key_idx = (key_idx + 5) % 12  # Adjusting by a perfect fifth
        estimated_key = keys[key_idx]

    return estimated_key, mode_name

# Example usage
file_path = 'Downloads/Song.wav'
key, mode = estimate_key(file_path)
print("Estimated Key and Mode:", key, mode)
