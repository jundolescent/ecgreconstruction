import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from Dataloader.ECGDataLoader import ECGDataLoader, min_max_scaling
from tqdm import tqdm
# input sequence: 1000
# output sequence: 1000

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(2)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class ECGReconstructor:
    def __init__(self, path, test_fold=10, batch_size=50, sampling_rate=100, lead=2):
        self.path = path
        self.sampling_rate = sampling_rate
        self.data_loader = ECGDataLoader(path, sampling_rate)
        self.test_fold = test_fold
        self.batch_size = batch_size
        self.lead = lead
        self.X_train, self.y_train, self.X_test, self.y_test = self.data_loader.preprocess_data(self.test_fold)
        self.X_train_scaled = min_max_scaling(self.X_train)
        self.X_test_scaled = min_max_scaling(self.X_test)
    def train(self):
        train_dataset = TensorDataset(torch.tensor(self.X_train_scaled, dtype=torch.float32),
                                      torch.tensor(self.y_train[:, :, self.lead], dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        input_size = 1
        # input_size = self.X_train.shape[1]
        hidden_size = 64
        num_layers = 4
        num_classes = input_size
        self.model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

        self.model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        num_epochs = 10

        for epoch in range(num_epochs):
            total_loss = 0

            with tqdm(train_loader, unit="batch") as tepoch:
                for inputs, targets in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    tepoch.set_postfix({'Loss': total_loss / len(tepoch)})
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

    def test(self):
        test_dataset = TensorDataset(torch.tensor(self.X_test_scaled, dtype=torch.float32),
                                      torch.tensor(self.y_test[:, :, self.lead], dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        input_size = self.X_test.shape[1]

        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            num_batches = 0
            for inputs, targets in test_loader:
                outputs = self.model(inputs)
                mse_loss = nn.MSELoss()
                loss = mse_loss(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

            average_loss = total_loss / num_batches
            print(f'Test Loss: {average_loss:.4f}')