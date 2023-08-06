from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from AudioProcessing import TacotronSTFT, get_mel
import os
import torch

class TextMelLoader():
    def __init__(self,configs,what = 'train'):
        assert what =='train' or what == 'test', "what argument accepts only train or test as arg str"
        
        if what=='train':
            df = pd.read_csv(configs['Training_Configs']['train_path'])
        else:
            df = pd.read_csv(configs['Training_Configs']['test_path'])
            
        self.tokenizer = AutoTokenizer.from_pretrained(configs['Text_Configs']['tokenizer_path'])
        
        self.input_text = df['sentence'].to_list()
        self.audio_path = df['audio_path'].to_list()
        self.sampling_rate = configs['Audio_Configs']['sampling_rate']
        
        self.stft = TacotronSTFT(filter_length=configs['Audio_Configs']['n_fft'], hop_length=configs['Audio_Configs']['hop_len'],sampling_rate=self.sampling_rate,win_length=configs['Audio_Configs']['win_len'])
        
    def __getitem__(self,idx):
        
        input_token = self.tokenizer.encode(self.input_text[idx],max_length = 50)
        out_mel = get_mel(self.stft,os.path.join(os.getcwd(),self.audio_path[idx]),sampling_rate=self.sampling_rate).transpose(0,1)
        
        
        return torch.tensor(input_token), out_mel
    
    def __len__(self):
        return len(self.input_text)
    
    
    def collate(self,batch):
        mel = []
        tokens = []
        gates = []
        
        for (input_token, out_mel) in batch:
            tokens.append(input_token)
            mel.append(out_mel)
            gates.append(torch.tensor([0 for _ in range(len(out_mel))]))
            
        out_mel = pad_sequence(mel,padding_value=0,batch_first=True)
        tokens = pad_sequence(tokens,padding_value=self.tokenizer.pad_token_id,batch_first=True)
        gates = pad_sequence(gates,batch_first=True,padding_value=1)
        
        return tokens, out_mel, gates
