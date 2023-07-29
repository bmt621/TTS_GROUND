import os
import torch
import librosa
import numpy as np
from typing import Union
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from scipy.io.wavfile import read
import librosa.util as librosa_util
from librosa.filters import mel as librosa_mel_fn
from scipy.signal import resample
from scipy.io import wavfile
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(win_length >= filter_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
        
class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=None):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output



def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


def resample_audio(data, orig_sr, target_sample_rate):
    # Read the audio file

    # Calculate the resampling ratio
    resampling_ratio = target_sample_rate / orig_sr
    
    # Perform resampling
    resampled_data = resample(data, int(len(data) * resampling_ratio))
    
    # Convert the sample rate to the target sample rate
    resampled_sample_rate = int(orig_sr * resampling_ratio)
    
    return resampled_data, resampled_sample_rate


def load_wav_to_torch(full_path, sr):
    sampling_rate, data = read(full_path)
    
    if sampling_rate != sr:
        data, _ = resample_audio(data,sampling_rate,sr)

    return torch.FloatTensor(data.astype(np.float32))


def get_mel(stft,filename,sampling_rate,max_wav_value=32768.0,load_mel_from_disk=False):
        if not load_mel_from_disk:
            audio = load_wav_to_torch(filename, sampling_rate)
            audio_norm = audio / max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), stft.n_mel_channels))

        return melspec


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


def add_silence_to_audio(
    audio_path: Union[os.PathLike, str], 
    output_path: Union[os.PathLike, str], 
    sample_rate: int=22050
):
    audio, sr = librosa.load(audio_path, sr=sample_rate)

    silence_sample_rate = 0.05 * sr
    silence_sample_rate = int(silence_sample_rate)

    silence = np.zeros(silence_sample_rate, dtype=audio.dtype)
    audio_with_silence = np.concatenate((silence, audio, silence), axis=0)

    sf.write(output_path, audio_with_silence, sr, format='wav')


def process_files_with_silence(
    audio_path_folder: Union[os.PathLike, str], 
    output_folder: Union[os.PathLike, str],
):
    with ThreadPoolExecutor() as executor:
        futures = []
        for audio in os.listdir(audio_path_folder):
            audio_path = os.path.join(audio_path_folder, audio)
            output_path = output_folder + "/" + audio_path.split("/")[-1]
           
            futures.append(executor.submit(add_silence_to_audio, audio_path, output_path))
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing audio: {e}")
