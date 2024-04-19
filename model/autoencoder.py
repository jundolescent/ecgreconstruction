import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Dataloader.ECGDataLoader import min_max_scaling, ECGDataLoader
from tqdm import tqdm
from utils.logger import JSLogger
import logging
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ECGAutoencoder:
    def __init__(self, input_size=1000, encoding_dim=128, learning_rate=0.001, batch_size=64, num_epochs=10, path='', test_fold=10, sampling_rate=100, level=logging.DEBUG, lead=2):
        self.test_fold = test_fold
        self.path = path
        self.sampling_rate = sampling_rate

        self.input_size = input_size
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Autoencoder(input_size, encoding_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.data_loader = ECGDataLoader(path, sampling_rate)
        self.X_train, self.y_train, self.X_test, self.y_test = self.data_loader.preprocess_data(self.test_fold)
        self.X_train_scaled = min_max_scaling(self.X_train)
        self.X_test_scaled = min_max_scaling(self.X_test)
        self.logger = JSLogger(level=level, log_file='../log.txt')
        self.lead = lead


    def train(self):
        train_dataset = TensorDataset(torch.tensor(self.X_train_scaled, dtype=torch.float32),
                                      torch.tensor(self.y_train[:, :, self.lead], dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            total_loss = 0
            with tqdm(train_loader, unit="batch") as tepoch:
                for inputs, targets in tepoch:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    tepoch.set_description(f"Epoch {epoch}")
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    tepoch.set_postfix({'Loss': total_loss / len(tepoch)})
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

    def reconstruct(self):
        test_input = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
        reconstructed = self.model(test_input)
        return reconstructed.cpu().detach().numpy()

    def test(self):
        test_dataset = TensorDataset(torch.tensor(self.X_test_scaled, dtype=torch.float32),
                                      torch.tensor(self.y_test[:, :, self.lead], dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        total_loss = 0

        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            num_batches = 0
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                mse_loss = nn.MSELoss()
                ### print for comparison
                print(outputs)
                print(targets)

                loss = mse_loss(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        average_loss = total_loss / num_batches
        print(f'Test Loss: {average_loss:.4f}')
        return average_loss



