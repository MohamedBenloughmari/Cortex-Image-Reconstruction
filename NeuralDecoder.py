import torch
from torch.utils.data import DataLoader , Dataset
from torch import nn
import torch.nn.functional as F
from ConvLSTM import ConvLSTM
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
id='1_01_02'

class CortexMnistDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        data = torch.load(path, mmap=True)
        self.images    = data['images']
        self.spikes_on = data['spikes_on'].unsqueeze(2)
        self.spikes_off= data['spikes_off'].unsqueeze(2)

    def __len__(self):
        return len(self.spikes_on)
    def __getitem__(self, idx):
        return (self.spikes_on[idx].clone(),
                self.spikes_off[idx].clone(),
                self.images[idx].clone())
    
class MnistNeuralDecoder(nn.Module):
    def __init__(self,grid_size : int):
        super().__init__()
        self.convlstm = ConvLSTM(
            img_size=(grid_size, grid_size),
            input_dim=1,
            hidden_dim=3,
            kernel_size=(11, 11), 
            batch_first=True,
            bidirectional=False,
            return_sequence=False,
            layer_norm=True,
            peephole=True)
        self.conv = nn.Conv2d(3,1, kernel_size=grid_size-27, padding=0)
    def forward(self, x):
        _,x,_ = self.convlstm(x)                                                      
        return self.conv(x[0])
    
def train(model, num_epochs: int = 5):

    Epochs = num_epochs
    criterion = torch.nn.MSELoss()
    optimizer = Adam(params=model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=1e-6)
    best_model = None
    best_loss = +torch.inf
    train_losses = []
    val_losses = []

    for epoch in range(Epochs):
        train_loss = 0.
        val_loss = 0.
        model.train()
        for spikes_on, _, images in tqdm(train_loader, desc=f'Epoch {epoch+1}/{Epochs}'):
            optimizer.zero_grad()
            spikes_on = spikes_on.float().to(device)
            images = images.float().to(device)
            out = model(spikes_on)
            out = out.squeeze(2)
            loss = criterion(out, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        model.eval()

        with torch.no_grad():
            for spikes_on, _, images in tqdm(test_loader, desc='Validation'):
                spikes_on = spikes_on.float().to(device)
                images = images.float().to(device)
                out_val = model(spikes_on)
                loss = criterion(out_val, images)
                val_loss += loss.item()

        if val_loss < best_loss:
            print('New model saved')
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())

        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}/{Epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Plot
    epochs_range = list(range(1, Epochs + 1))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.suptitle('Training Curves', fontsize=14, fontweight='bold')
    ax1.plot(epochs_range, train_losses, 'o-', color='steelblue', label='Train Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1b = ax1.twinx()
    ax1b.plot(epochs_range, val_losses, 's--', color='tomato', label='Val Loss', linewidth=2)
    ax1b.set_ylabel('Val Loss', color='tomato')
    ax1b.tick_params(axis='y', labelcolor='tomato')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.set_xticks(epochs_range)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'training_curves{id}.png', dpi=150, bbox_inches='tight')
    plt.show()

    return best_model




if __name__=="__main__":
    train_dataset = CortexMnistDataset(f'cortex_mnist{id}/train.pt')
    val_dataset   = CortexMnistDataset(f'cortex_mnist{id}/val.pt')
    test_dataset  = CortexMnistDataset(f'cortex_mnist{id}/test.pt')

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=10, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=10, shuffle=False, num_workers=0)

    device=torch.device('mps')

    model=MnistNeuralDecoder(grid_size=40)
    model.to(device)
    best_model=train(model,num_epochs=5)
    torch.save(best_model,f'NeuralDecoder{id}.pth')