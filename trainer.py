from tts_transformers import *
from model_utils import *
from utils import load_configs
from textmel_loader import TextMelLoader
from torch.utils.data import DataLoader
from losses import TTS_Loss
import pytorch_lightning as pl
import numpy as np


class TTS_Model(pl.LightningModule):
    
    def __init__(self):
        super(TTS_Model,self).__init__()
        
        self.configs = load_configs('hparams.yaml')
        self.batch_size = self.configs['Training_Configs']['train_batch_size']
        self.max_epochs = self.configs['Training_Configs']['epoch']
        self.input_embed = nn.Embedding(self.configs['EncDec_Configs']['vocab_size'],self.configs['EncDec_Configs']['embed_dim'])
        
        self.wpe = PositionalEncoding(self.configs['EncDec_Configs']['embed_dim'],self.configs['EncDec_Configs']['dropout'],max_len=self.configs['EncDec_Configs']['max_length'])
        
        enc_layer = TransformerEncoderLayer(self.configs['EncDec_Configs']['embed_dim'],self.configs['EncDec_Configs']['n_head'],
                                            self.configs['EncDec_Configs']['d_ff'],self.configs['EncDec_Configs']['dropout'],batch_first=True)
        
        dec_layer = TransformerDecoderLayer(self.configs['EncDec_Configs']['embed_dim'],self.configs['EncDec_Configs']['n_head'],
                                            self.configs['EncDec_Configs']['d_ff'],self.configs['EncDec_Configs']['dropout'],batch_first=True)
        self.encode = TransformerEncoder(enc_layer,num_layers=self.configs['EncDec_Configs']['n_encoder_layer'])
        self.decode = TransformerDecoder(dec_layer,self.configs['EncDec_Configs']['n_decoder_layer'])
        
        self.encoder_prenet = EncoderPrenet(self.configs['EncDec_Configs']['embed_dim'])
        self.decoder_prenet = DecoderPrenet(self.configs['Audio_Configs']['num_mels'],self.configs['EncDec_Configs']['embed_dim'])
        self.start = nn.Embedding(1,self.configs['Audio_Configs']['num_mels'])
        
        self.head = HeadPredictor(self.configs,self.configs['EncDec_Configs']['dropout'])
        self.loss_fn = TTS_Loss(weight=self.configs['Training_Configs']['weight'])
        
        
    def forward(self,src,mel,tgt_padding_mask=None):
        """pass input tokens and mel spectrograms

        Args:
            src (_type_): text input of shape (B, T)
            mel (_type_): mel spectrogram of shape (B, T, n_mel)
            mel_mask (_type_): _description_
            src_key_padding_mask (_type_): _description_
            
        """
        
        memory = self.encoder(src)
        
        mel_before, mel_final, gate = self.decoder(mel,memory,tgt_padding_mask)
        
        return mel_before, mel_final, gate
    
    def parse_mel_and_padding_mask(self,mel,tgt_padding_mask):
        
        start_mel = self.start(torch.tensor([0]).to(self.device)).unsqueeze(1).expand([mel.shape[0],1,mel.shape[-1]]) # Get the start Embeddings
        
        mel = torch.cat([start_mel,mel],dim=1)
        
        tgt_padding_mask = torch.cat([torch.tensor(0).unsqueeze(0).expand(tgt_padding_mask.shape[0],1).to(self.device),tgt_padding_mask],dim=1)
        
        return mel, tgt_padding_mask
    
    def training_step(self,batch,batch_idx):
        input_id, output_mel, mel_padding = batch
        output_mel, mel_padding = self.parse_mel_and_padding_mask(output_mel,mel_padding)
        
        mel_input = output_mel[:,:-1,:]
        mel_output = output_mel[:,1:,:]
        mel_padding = mel_padding[:,:-1]

        pred = self.forward(input_id,mel_input,mel_padding)
        target = (mel_output,mel_padding)
                
        loss = self.loss_fn(pred,target)

        self.log("train_loss",loss,prog_bar=True,logger=True,on_step=True,on_epoch=True)

        return loss
    
    def validation_step(self,batch,batch_idx):
        
        input_id, output_mel, mel_padding = batch
        output_mel, mel_padding = self.parse_mel_and_padding_mask(output_mel,mel_padding)
        
        mel_input = output_mel[:,:-1,:]
        mel_output = output_mel[:,1:,:]
        mel_padding = mel_padding[:,:-1]

        pred = self.forward(input_id,mel_input,mel_padding)
        target = (mel_output,mel_padding)
                
        loss = self.loss_fn(pred,target)
        
        metrics = {'val_loss':loss}

        self.log_dict(metrics,prog_bar=True,on_step=True,on_epoch=True,logger=True)

        return loss
    
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(),lr=self.configs['Training_Configs']['lr'],betas=(0.9, 0.98), eps=1e-9)
        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch

        # Set warmup to 10% of total training iterations (adjust as needed)
        warmup_ratio = 0.05
        self.total_train_iterators = (self.batch_size * self.max_epochs)
        
        warmup_steps = int(warmup_ratio * self.total_train_iterators)

        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=warmup_steps, max_iters=self.total_train_iterators
        )
        return optimizer
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration


    """def configure_gradient_clipping(self, optimizer,gradient_clip_val=1, gradient_clip_algorithm="norm"):
        self.clip_gradients(
            optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm
        )"""
    
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
    
    def decoder(self,mel, memory,tgt_padding_mask):
        """ 
        the decoder receives mel spectrograms (B, T, n_mel) an memory of the encoder layers,
        the docoder then passes it to decoder layer and head layer to predict the spectrograms,
        and the gate output.

        Args:
            mel (_type_): mel spectrogram (B, T, n_mel)
            mel_mask (_type_): 
            memory  (_type_): memory of encoder (B, T, n_hid)
            
        """
        
        if tgt_padding_mask != None:
            
            tgt_padding_mask = tgt_padding_mask.float().to(self.device)
            
        _, tgt_t, dim = mel.shape
        
        self.register_buffer('attn_mask',torch.triu(torch.full((tgt_t,tgt_t), float('-inf')).to(self.device),diagonal=1))
        
        
        mel_in = self.decoder_prenet(mel)
        
        mel_in = self.wpe(mel_in)
        
        output = self.decode(mel_in, memory, tgt_mask = self.attn_mask,tgt_key_padding_mask=tgt_padding_mask)
        
        mel_before, mel_final, gate = self.head(output)
        
        return mel_before, mel_final, gate
        
        
    def make_src_pad_mask(self,src,pad_idx):
        src_padding = (src == pad_idx).float()
        return src_padding
    
    def save_tts_checkpoint(self,checkpoint_path):
        checkpoint = {'state_dict':self.state_dict()}
        torch.save(checkpoint,checkpoint_path)
        print("successfully saved state dict to path {}".format(checkpoint_path))
        
    def load_tts_callback(self,checkpoint_path):
        
        checkpoint = torch.load(checkpoint_path)
        st_dict = checkpoint['state_dict']
        st_dict.pop('attn_mask')
        
        self.load_state_dict(st_dict)
        print("succesfully loaded state dict from path {}".format(checkpoint_path))
        

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def main():
        
    configs = load_configs('hparams.yaml')
    
    train_ds = TextMelLoader(configs,what='train')
    test_ds = TextMelLoader(configs,what= 'test')
    
    train_loader = DataLoader(train_ds,batch_size=configs['Training_Configs']['train_batch_size'],shuffle=True,collate_fn=train_ds.collate)
    test_loader = DataLoader(test_ds,batch_size=configs['Training_Configs']['test_batch_size'],shuffle=False,collate_fn=test_ds.collate)
        
    model = TTS_Model()
    
    trainer = pl.Trainer(default_root_dir='callbacks',accelerator='gpu',devices=[2],max_epochs=1000,gradient_clip_val=1,gradient_clip_algorithm='norm',detect_anomaly=False)
    #model.load_tts_checkpoint('tts1.pt')
    ckpt = torch.load('tts6.pt')['state_dict']
    ckpt.pop('attn_mask')
    model.load_state_dict(ckpt)
    try:
        
        trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=test_loader)
    
    except (Exception, KeyboardInterrupt) as e:
        
        model.save_tts_checkpoint("tts6.pt")
    
if __name__=="__main__":
    main()