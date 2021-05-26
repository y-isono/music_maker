from os.path import join

from sklearn.neural_network import MLPRegressor, MLPClassifier
from pydub import AudioSegment
from utils import get_mfccs
from utils import compress_audio_segment
import numpy as np

# Preload NN models using pickle
import pickle

import argparse
nn_code_to_genre_map = {
	0: "Western Classical",
	1: "East Asia Classical",
	2: "Blues",
	3: "Country",
	4: "Disco",
	5: "Hiphop",
	6: "Jazz",
	7: "Metal",
	8: "Pop",
	9: "Rock",
	10: "Electronic",
	11: "New Age",
	12: "Soundtracks"
}

# from https://github.com/muchen2/DeepSent
class DeepSent:
    pace_regressor = None
    arousal_regressor = None
    valence_regressor = None
    genre_classifier = None
    version_code = "v2"
    
    def __init__(self, model_path):
        print('DeepSent initialize')
        with open(join(model_path, "nnet_models/pace_regressor_" + self.version_code + ".pickle"), "rb") as f:
	        self.pace_regressor = pickle.load(f)
        with open(join(model_path, "nnet_models/arousal_regressor_" + self.version_code + ".pickle"), "rb") as f:
	        self.arousal_regressor = pickle.load(f)
        with open(join(model_path, "nnet_models/valence_regressor_" + self.version_code + ".pickle"), "rb") as f:
	        self.valence_regressor = pickle.load(f)
        with open(join(model_path, "nnet_models/genre_classifier_" + self.version_code + ".pickle"), "rb") as f:
	        self.genre_classifier = pickle.load(f)

    def music2sent(self, wav_file_path, file_name):
        print('call DeepSent.music2sent')
        aud = AudioSegment.from_wav(wav_file_path)
        aud_em = compress_audio_segment(aud, 11025, 1)
        aud_gn = compress_audio_segment(aud, 22050, 1)
        wave_data_em = np.asarray(aud_em.get_array_of_samples())
        wave_data_gn = np.asarray(aud_gn.get_array_of_samples())

        # Convert the middle 50% part of the music into MFCC arrays
        # which will be fed into the regressors and the genre classifier
        frame_length = 5000
        frame_step = 500
        mfcc_frame_length = 25
        num_mfcc_coef_kept = 12
        # compressed format always have sample rate of 11025
        frame_length_i = int(frame_length / 1000. * 11025)
        frame_step_i = int(frame_step / 1000. * 11025)

        first_quarter = int(len(wave_data_em) * 0.25)
        last_quarter = int(len(wave_data_em) * 0.75)

        if (last_quarter - first_quarter) / 11025. < 5.0:
            print('given music is too short')
            quit()
            # Middle section is less than 5 seconds
	        # Music is too short for analysis

        mid_segment = wave_data_em[first_quarter:last_quarter]
        num_frame = (len(mid_segment) - frame_length_i) // frame_step_i


        mfccs_mat = np.zeros(
            (num_frame, int(frame_length / mfcc_frame_length * num_mfcc_coef_kept)))
        for i in range(num_frame):
            start_pos = i * frame_step_i
            end_pos = start_pos + frame_length_i
            mfccs = get_mfccs (mid_segment[start_pos:end_pos], sample_rate=11025, frame_length=mfcc_frame_length,
            frame_step=mfcc_frame_length, num_coef_kept=num_mfcc_coef_kept)
            mfccs_mat[i] = mfccs.flatten()

        # feed the data into regressors
        pace_regressor_result = self.pace_regressor.predict (mfccs_mat) + 1.0
        arousal_regressor_result = self.arousal_regressor.predict (mfccs_mat) + 1.5
        valence_regressor_result = self.valence_regressor.predict (mfccs_mat) + 1.5

        # shrink overrated scores
        pace_regressor_result[np.where (pace_regressor_result>2.0)[0]] = 2.0
        pace_regressor_result[np.where (pace_regressor_result<0.0)[0]] = 0.0
        arousal_regressor_result[np.where(arousal_regressor_result>3.0)[0]] = 3.0
        arousal_regressor_result[np.where(arousal_regressor_result<0.0)[0]] = 0.0
        valence_regressor_result[np.where(valence_regressor_result>3.0)[0]] = 3.0
        valence_regressor_result[np.where(valence_regressor_result<0.0)[0]] = 0.0

        # calculate mean results
        pace_score = np.mean (pace_regressor_result)
        arousal_score = np.mean (arousal_regressor_result)
        valence_score = np.mean (valence_regressor_result)

        # calculate result ratios
        pace_fast_ratio = len (np.where (pace_regressor_result>1.33)[0]) / float (len (pace_regressor_result))
        pace_slow_ratio = len (np.where (pace_regressor_result<0.66)[0]) / float (len (pace_regressor_result))
        pace_mid_ratio = 1. - pace_fast_ratio - pace_slow_ratio

        arousal_intense_ratio = len (np.where (arousal_regressor_result>2.0)[0]) / float (len (arousal_regressor_result))
        arousal_relaxing_ratio = len (np.where (arousal_regressor_result<1.0)[0]) / float (len (arousal_regressor_result))
        arousal_mid_ratio = 1. - arousal_intense_ratio - arousal_relaxing_ratio

        valence_happy_ratio = len (np.where (valence_regressor_result>2.0)[0]) / float (len (valence_regressor_result))
        valence_sad_ratio = len (np.where (valence_regressor_result<1.0)[0]) / float (len (valence_regressor_result))
        valence_neutral_ratio = 1. - valence_happy_ratio - valence_sad_ratio

        # now classify the music genre
        first_quarter = int (len(wave_data_gn) * 0.25)
        last_quarter = int (len(wave_data_gn) * 0.75)
        mid_segment = wave_data_gn[first_quarter:last_quarter]
        mfccs_mid_segment = get_mfccs (mid_segment, sample_rate=22050, frame_length=20, frame_step=20, n_filters=20, num_coef_kept=15)
        mfccs_mean = np.mean (mfccs_mid_segment, axis=0)
        triu_indices = np.triu_indices (len (mfccs_mean))
        cov_mat = np.cov (mfccs_mid_segment.T)
        mfccs_cov_mat_upper_flatten = cov_mat[triu_indices]
        data_mid_segment = np.concatenate ((mfccs_mean, mfccs_cov_mat_upper_flatten))

        genre_probs = self.genre_classifier.predict_proba ([data_mid_segment])[0]
        sorted_indices = np.argsort (genre_probs)[::-1]
        best_cand = nn_code_to_genre_map[sorted_indices[0]]
        sec_best_cand = nn_code_to_genre_map[sorted_indices[1]]

        # return results in json
        result_dict = {
	        "filename": file_name, 
	        "pace_score": pace_score / 2.0 * 100.0,
	        "arousal_score": arousal_score / 3.0 * 100.0,
	        "valence_score": valence_score / 3.0 * 100.0,
	        "pace_fast_ratio": pace_fast_ratio * 100.0,
	        "pace_slow_ratio": pace_slow_ratio * 100.0,
	        "pace_mid_ratio": pace_mid_ratio * 100.0,
	        "arousal_intense_ratio": arousal_intense_ratio * 100.0,
	        "arousal_relaxing_ratio": arousal_relaxing_ratio * 100.0,
	        "arousal_mid_ratio": arousal_mid_ratio * 100.0,
	        "valence_happy_ratio": valence_happy_ratio * 100.0,
	        "valence_sad_ratio": valence_sad_ratio * 100.0,
	        "valence_neutral_ratio": valence_neutral_ratio * 100.0,
	        "best_cand": best_cand,
	        "sec_best_cand": sec_best_cand
        }

        return result_dict


def test(model_path, file_path, file_name):
    deep_sent = DeepSent(model_path=model_path)
    result_dict = deep_sent.music2sent(file_path, file_name)
    print(result_dict)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', type=str, default="./")
    parser.add_argument('--file_path', type=str, default="./test2.wav")
    parser.add_argument('--file_name', type=str, default="test")
    args = parser.parse_args()
    model_path = args.model_path
    file_path = args.file_path
    file_name = args.file_name
    test(model_path, file_path, file_name)