import os
import random
import numpy as np
from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity

# Reproducible results.
np.random.seed(123)
random.seed(123)

# Define the model here.
model = DeepSpeakerModel()

# Load the checkpoint.
model.m.load_weights('ResCNN_triplet_training_checkpoint_265.h5', by_name=True)

# Get WAV file paths from environment variables.
wav_file_1 = os.getenv('WAV_FILE_1', 'samples/user1.mp3')  # Default to 'samples/1.wav'
wav_file_2 = os.getenv('WAV_FILE_2', 'samples/user2.mp3')  # Default to 'samples/5.wav'

# Sample some inputs for WAV/FLAC files for the same speaker.
mfcc_001 = sample_from_mfcc(read_mfcc(wav_file_1, SAMPLE_RATE), NUM_FRAMES)
mfcc_002 = sample_from_mfcc(read_mfcc(wav_file_2, SAMPLE_RATE), NUM_FRAMES)

# Call the model to get the embeddings of shape (1, 512) for each file.
predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))

# Do it again with a different speaker.
mfcc_003 = sample_from_mfcc(read_mfcc('samples/1255-90413-0001.flac', SAMPLE_RATE), NUM_FRAMES)
predict_003 = model.m.predict(np.expand_dims(mfcc_003, axis=0))

# Compute the cosine similarity and check that it is higher for the same speaker.
same_speaker_similarity = batch_cosine_similarity(predict_001, predict_002)
diff_speaker_similarity = batch_cosine_similarity(predict_001, predict_003)
print('SAME SPEAKER', same_speaker_similarity)  # SAME SPEAKER [0.81564593]
print('DIFF SPEAKER', diff_speaker_similarity)  # DIFF SPEAKER [0.1419204]

assert same_speaker_similarity > diff_speaker_similarity
