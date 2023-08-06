from tts_transformers import *
from model_utils import *
from utils import load_configs
from textmel_loader import TextMelLoader
from torch.utils.data import DataLoader

class TTS_Model(nn.Module):
    
    def __init__(self,configs,device):
        super(TTS_Model,self).__init__()
        
        self.configs = configs
        self.device = device
        self.input_embed = nn.Embedding(configs['EncDec_Configs']['vocab_size'],configs['EncDec_Configs']['embed_dim'])
        
        self.wpe = PositionalEncoding(configs['EncDec_Configs']['embed_dim'],configs['EncDec_Configs']['dropout'],max_len=configs['EncDec_Configs']['max_length'])
        
        enc_layer = TransformerEncoderLayer(configs['EncDec_Configs']['embed_dim'],configs['EncDec_Configs']['n_head'],
                                            configs['EncDec_Configs']['d_ff'],configs['EncDec_Configs']['dropout'],batch_first=True)
        
        dec_layer = TransformerDecoderLayer(configs['EncDec_Configs']['embed_dim'],configs['EncDec_Configs']['n_head'],
                                            configs['EncDec_Configs']['d_ff'],configs['EncDec_Configs']['dropout'],batch_first=True)
        self.encode = TransformerEncoder(enc_layer,num_layers=configs['EncDec_Configs']['n_encoder_layer'])
        self.decode = TransformerDecoder(dec_layer,configs['EncDec_Configs']['n_decoder_layer'])
        
        self.encoder_prenet = EncoderPrenet(configs['EncDec_Configs']['embed_dim'])
        self.decoder_prenet = DecoderPrenet(configs['Audio_Configs']['num_mels'],configs['EncDec_Configs']['embed_dim'])
        
        self.head = HeadPredictor(configs,configs['EncDec_Configs']['dropout'])
        
        
    def forward(self,src,mel,tgt_padding_mask=None):
        """pass input tokens and mel spectrograms

        Args:
            src (_type_): text input of shape (B, T)
            mel (_type_): mel spectrogram of shape (B, T, n_mel)
            mel_mask (_type_): _description_
            src_key_padding_mask (_type_): _description_
            
        """
        
        memory = self.encoder(src)
        
        mel_before, mel_final, gate = self.decoder(mel,memory)
        
        return mel_before, mel_final, gate
    
    def encoder(self,src):
        """ pass input token to produce memory

        Args:
            src (_type_): mel spectrogram (B, T)
            
        """
        
        _, src_t = src.shape

        src_padding = self.make_src_pad_mask(src,self.configs['EncDec_Configs']['pad_idx']).to(self.device)
        src_tok_emb = self.input_embed(src)
        
        src_tok_emb = self.encoder_prenet(src_tok_emb)
        src_tok_emb = self.wpe(src_tok_emb)

        memory = self.encode(src_tok_emb,src_key_padding_mask=src_padding)

        return memory
    
    def decoder(self,mel, memory):
        """ 
        the decoder receives mel spectrograms (B, T, n_mel) an memory of the encoder layers,
        the docoder then passes it to decoder layer and head layer to predict the spectrograms,
        and the gate output.

        Args:
            mel (_type_): mel spectrogram (B, T, n_mel)
            mel_mask (_type_): 
            memory  (_type_): memory of encoder (B, T, n_hid)
            
        """
        
        _, tgt_t, dim = mel.shape
        
        self.register_buffer('attn_mask',torch.triu(torch.full((tgt_t,tgt_t), float('-inf')).to(self.device),diagonal=1))
        
        """if tgt_padding_mask!=None:
            tgt_padding_mask = (tgt_padding_mask).float().to(self.device)"""
        
        mel_in = self.decoder_prenet(mel)
        mel_in = self.wpe(mel_in)
        
        output = self.decode(mel_in, memory, tgt_mask = self.attn_mask)
        
        mel_before, mel_final, gate = self.head(output)
        
        return mel_before, mel_final, gate
        
    
    def make_src_pad_mask(self,src,pad_idx):
        src_padding = (src == pad_idx).float()
        return src_padding

def main():
    
    configs = load_configs('hparams.yaml')
    device = torch.device('cuda:1')
    
    model = TTS_Model(configs,device)
    model.to(device)
    
    configs = load_configs('hparams.yaml')
    train_ds = TextMelLoader(configs,what='train')
    
    train_loader = DataLoader(train_ds,batch_size=10,shuffle=True,collate_fn=train_ds.collate)
    batch = next(iter(train_loader))
    
    input_id, out_mel, gate = batch
    input_id = input_id.to(device)
    out_mel = out_mel.to(device)
    gate = gate.to(device)
    
    output = model(input_id,out_mel,gate)
    
    print("output shape: ",output.shape)
    

if __name__=="__main__":
    main()