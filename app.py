import wave
import pyaudio
from doctest import OutputChecker
from bs4 import ResultSet
from flask import Flask, render_template, request
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import keras.utils
from keras.utils import custom_object_scope
from keras.backend import ctc_batch_cost

# User defing loss


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(
        y_true, y_pred, input_length, label_length)
    return loss


# loading the Model 
model = load_model('models/notebook22.h5', custom_objects={'CTCLoss': CTCLoss})
model.compile(loss=CTCLoss, optimizer='adam')

# Recording and saving part

app = Flask(__name__)


@app.route('/')
def index(): 
    return render_template('index.html')
@app.route('/about')
def about(): 
    return render_template('about.html')
@app.route('/team')
def team():
    return render_template('team.html')


@app.route('/record', methods=['POST'])
def record():
    filename = "wavefile/LJ001-0005.wav"
    chunk = 1024
    FORMAT = pyaudio.paInt16
    channels = 1
    sample_rate = 44100
    record_seconds = 10

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []
    print("Recording....")
    for i in range(int(44100 / chunk*record_seconds)):
        data = stream.read(chunk)
        # stream.write(data)
        frames.append(data)
    print("Finished recording..")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()
    return 'success'

    # csv file made
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save('wavefile/LJ001-0005.wav')
    return render_template('index.html', success = "Successfully Added !")


@app.route('/result', methods=['POST'])
def result():
    import csv
    fields = ['file_name', 'transcription']
    rows = [['LJ001-0005', ' a']]
    filename = "record.csv"
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

    # csv file manupulating
    metadata_df = pd.read_csv(filename, header=None)
    metadata_df.columns = ['file_name', 'transcription']
    metadata_df = metadata_df[metadata_df.file_name != 'file_name']

    # defining
    df_validate = metadata_df

    # char_to_num and num_to_char conversion
    char = [x for x in "abcdefghijklmnopqrstuvwxyz',.?! "]
    char_to_num = keras.layers.StringLookup(vocabulary=char, oov_token="")
    num_to_char = keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

    # feature extraction
    frame_length = 256
    frame_step = 160
    fft_length = 384
    wave_data = 'wavefile/'

    def encode_sample(wave_file, label):
        file = tf.io.read_file(wave_data + wave_file + '.wav')
        audio, _ = tf.audio.decode_wav(file)
        audio = tf.squeeze(audio, axis=1)
        audio = tf.cast(audio, tf.float32)

        spectogram = tf.signal.stft(
            audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
        )
        spectogram = tf.abs(spectogram)
        spectogram = tf.math.pow(spectogram, 0.5)

        means = tf.math.reduce_mean(spectogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectogram, 1, keepdims=True)
        spectogram = (spectogram - means) / (stddevs + 1e-10)
        label = tf.strings.lower(label)
        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        label = char_to_num(label)
        return spectogram, label

    # data mapping
    batch_size = 32
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_validate["file_name"]), list(df_validate["transcription"])))

    validation_dataset = (
        validation_dataset.map(
            encode_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    # prediction function

    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(
            pred, input_length=input_len, greedy=True)[0][0]
        output_text = []
        for result in results:
            result = tf.strings.reduce_join(
                num_to_char(result)).numpy().decode("utf-8")
            output_text.append(result)
        return output_text

    # Result Predictions
    predictions = []
    targets = []
    for batch in validation_dataset:
        X, y = batch

        batch_predictions = model.predict(X)
        batch_predictions = decode_batch_predictions(batch_predictions)
        predictions.extend(batch_predictions)
    # for i in np.random.randint(0, len(predictions), 1):
    #     print(f"Prediction: {predictions[i]}")
    result = predictions[0]
    print(result)
    print("." * 100)
    output = result
    return render_template('index.html', output=output)
 
 
if __name__ == "__main__":
    app.run(debug=True)
