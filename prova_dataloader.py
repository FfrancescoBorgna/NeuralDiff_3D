import torch
from torch.utils.data import DataLoader, Dataset

# Define your custom dataset
class CustomDataset(Dataset):
    def __init__(self, your_data):
        self.data = your_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return torch.tensor(sample, dtype=torch.float32)

# Instantiate the custom dataset
your_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9],[10,11,12]]  # Replace this with your actual data
custom_dataset = CustomDataset(your_data)

# Create a DataLoader with batch_size=1 to access samples sequentially
data_loader = DataLoader(dataset=custom_dataset, batch_size=3, shuffle=False)

# Iterate through the DataLoader to access samples sequentially
for batch in data_loader:
    sample = batch.squeeze()  # Remove the batch dimension if present
    print("Sample:", sample)
