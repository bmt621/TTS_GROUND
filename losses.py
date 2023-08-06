import torch.nn as nn
from torch import tensor

class TTS_Loss(nn.Module):
    def __init__(self,weight):
        super(TTS_Loss,self).__init__()
        self.mel_linear_loss = nn.MSELoss()
        self.post_mel_loss = nn.MSELoss()

        self.gate_loss_fn = nn.BCEWithLogitsLoss(pos_weight=tensor(weight))

    def forward(self,predicted,target):

        mel_linear_out, post_mel_out, gate_out = predicted
        mel_target, gate_target = target[0],target[1]

        post_loss = self.post_mel_loss(post_mel_out,mel_target)
        mel_after = self.mel_linear_loss(mel_linear_out,mel_target)

        mel_loss = post_loss + mel_after

        gate_loss = self.gate_loss_fn(gate_out.reshape(-1,1),gate_target.reshape(-1,1).float())

        total_loss = mel_loss + gate_loss

        return total_loss