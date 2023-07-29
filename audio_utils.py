from librosa.filters import mel as librosa_mel_fn
from scipy.signal import resample
from scipy.io import wavfile
import os
import numpy as np

def resample_audio(input_file, output_file, target_sample_rate):
    # Read the audio file
    sample_rate, data = wavfile.read(input_file)
    
    # Calculate the resampling ratio
    resampling_ratio = target_sample_rate / sample_rate
    
    # Perform resampling
    resampled_data = resample(data, int(len(data) * resampling_ratio))
    
    # Convert the sample rate to the target sample rate
    resampled_sample_rate = int(sample_rate * resampling_ratio)
    
    # Write the resampled audio to a new file
    wavfile.write(output_file, resampled_sample_rate, resampled_data.astype(data.dtype))


def export_to_signed_16_bit(input_file, output_file):
    # Read the audio file
    sample_rate, data = wavfile.read(input_file)
    
    # Check if the audio data is already in signed 16-bit format
    if data.dtype == np.int16:
        print("The input file is already in signed 16-bit format.")
    
    # Convert the audio data to signed 16-bit format
    data_signed_16_bit = data.astype(np.int16)
    
    # Write the audio data to a new file
    wavfile.write(output_file, sample_rate, data_signed_16_bit)

def resample_and_export(input_file,output_file,target_sr):
    #load the audio
    sr, data = wavfile.read(input_file)
    r_ratio = target_sr/sr

    if r_ratio!=1:
        new_data = resample(data,int(len(data)*r_ratio))
        if new_data.dtype!=np.int16:
            new_data = new_data.astype(np.int16)
        wavfile.write(output_file,target_sr,new_data)
    else:
        if data.dtype!=np.int16:
            data = data.astype(np.int16)
        wavfile.write(output_file,sr,data)



def resample_and_export_all_audio(audio_paths,new_base_audio,sr=22050):
    all_files = os.listdir(audio_paths)
    for files in all_files:
        wav_path = os.path.join(audio_paths,files)
        out_path = os.path.join(new_base_audio,files)
        resample_and_export(wav_path,out_path,sr)
    print("done resampling...")
    