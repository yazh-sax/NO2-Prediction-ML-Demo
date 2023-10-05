import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

# Predict Nitrogen Dioxide (NO2) levels from other pollutant levels

writer = SummaryWriter("runs/aqi")

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
input_size = 7  # No. of input values
hidden_size = 7  # size of hidden layers
output_size = 1  # output layer size
num_epochs = 100 # training iterations
learning_rate = 0.001  # model learning rate
batch_size = 10  # training input/label pairs per batch


# Load Data
class CSVDataset(Dataset):
    def __init__(self, file_name, data_index, label_index):
        # Load X/y Data from .csv file
        file_out = pd.read_csv(file_name)

        y = file_out[label_index]
        x = file_out[data_index]

        # Scale features and assign to attributes
        sc = StandardScaler()
        self.X = sc.fit_transform(x)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x_train = torch.tensor(self.X[item], dtype=torch.float32)
        y_train = torch.tensor(self.y.iloc[item], dtype=torch.float32)
        return x_train, y_train




dataset = CSVDataset("delhi_aqi.csv", ["co", "no", "o3", "so2", "pm2_5", "pm10", "nh3"], ["no2"])

# Splitting into train/test and creating iterable object
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


# Model
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


# Model Implementation
model = NeuralNet(input_size, hidden_size, output_size)
model = model.to(device)


# Loss and Optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
total_steps = len(train_loader)

running_loss = 0.0


for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)


        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()


        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1}/{total_steps}, loss = {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss / 100, epoch * total_steps + i)
            running_loss = 0.0
