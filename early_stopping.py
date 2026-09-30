import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        """
        patience: How many epochs to wait for an improvement before stopping.
        min_delta: Minimum change in the loss to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0 # Reset counter because loss improved

    def save_checkpoint(self, model):
        '''Saves model when validation loss decreases.'''
        torch.save(model.state_dict(), 'best_model.pth')
        print("Validation loss improved. Saved model weights to disk!")
