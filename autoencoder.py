import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=256):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class Autoencoder_layer3(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim1=1024, hidden_dim2=512, output_dim=256):
        super(Autoencoder_layer3, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_autoencoder(fingerprints, epochs=125, batch_size=256, learning_rate=0.001, autoencoder_type='layer2'):
    torch.manual_seed(2)
    if autoencoder_type == 'layer3':
        autoencoder = Autoencoder_layer3()
    else:
        autoencoder = Autoencoder()

    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    fingerprints_tensor = torch.tensor(fingerprints, dtype=torch.float32)
    dataset = TensorDataset(fingerprints_tensor, fingerprints_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for data in dataloader:
            inputs, _ = data
            optimizer.zero_grad()
            encoded, decoded = autoencoder(inputs)
            loss = criterion(decoded, inputs)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    return autoencoder.encoder
