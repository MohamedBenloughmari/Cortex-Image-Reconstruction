import torch
from torch.utils.data import DataLoader , Dataset
from torch import nn
import torch.nn.functional as F
from ConvLSTM import ConvLSTM
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import copy

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


def train(model,num_epochs:int=5):
    Epochs = num_epochs
    criterion = torch.nn.MSELoss()
    optimizer = Adam(params=model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=1e-6)
    best_model = None
    best_loss = +torch.inf
    for epoch in range(Epochs):
        train_loss = 0.
        val_loss = 0.
        model.train()
        for spikes_on, _, images in tqdm(train_loader, desc=f'Epoch {epoch+1}/{Epochs}'):
            optimizer.zero_grad()
            spikes_on = spikes_on.float().to(device)
            images = images.float().to(device)
            out = model(spikes_on)
            out=out.squeeze(2)
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

    return best_model



if __name__=="__main__":
    train_dataset = CortexMnistDataset('cortex_mnist/train.pt')
    val_dataset   = CortexMnistDataset('cortex_mnist/val.pt')
    test_dataset  = CortexMnistDataset('cortex_mnist/test.pt')

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=10, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=10, shuffle=False, num_workers=0)

    device=torch.device('mps')

    model=MnistNeuralDecoder(grid_size=40)
    model.to(device)
    best_model=train(model,num_epochs=5)
    torch.save(best_model,'NeuralDecoder.pth')