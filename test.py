from model import TTS_Model
from torch.utils.data import DataLoader
from textmel_loader import TextMelLoader
from losses import TTS_Loss
import pytorch_lightning as pl



class Trainer(pl.LightningModule):
    def __init__(self,configs):
        super(Trainer,self).__init__()
        
        self.configs= configs
        self.model = TTS_Model(configs,self.device)
        self.loss = TTS_Loss(weight=5)
        
    def forward(self):
        pass
    
    def training_step(self,batch,batch_idx):
        input_id, output_id = batch

        tgt_input = output_id[:,:-1]
        tgt_output = output_id[:,1:]

        logits = self(input_id,tgt_input)

        loss = self.loss_fn(logits.reshape(-1,logits.shape[-1]),tgt_output.reshape(-1))


        self.log("train_loss",loss,prog_bar=True,logger=True,on_step=True,on_epoch=True)

        return loss    