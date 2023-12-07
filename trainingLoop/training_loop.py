import torch
import torch.nn as nn
import math
from livelossplot import PlotLosses

class TrainingLoop:
    def __init__(
        self, 
        model: nn.Module,
        criterion: nn.modules.loss,
        optimizer: torch.optim,
        ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(
        self, 
        dataloader: torch.utils.data.dataloader,
        device: torch.device,
        num_epochs: int = 10
        ):
        self.model.train()
        self.model.to(device)
        
        for epoch in range(num_epochs):
            logs={}
            total_loss = 0

            for batch in dataloader:
                batch = batch.to(device)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = outputs.transpose(1,2)
                outputs = outputs[:, :, :targets.size(1)]
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logs['loss'] = total_loss/len(dataloader)
            logs['perplexity'] = math.exp(logs['loss'])
            plotlosses.update(logs)
            plotlosses.send()
        
        average_loss = total_loss / len(dataloader)
        final_perp = math.exp(average_loss)
        return final_perp