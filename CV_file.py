import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer
from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins.hparams import api as hp



# Load the dataset
data = pd.read_csv('Resume\Resume.csv')

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['Resume_str'], data['Category'], test_size=0.2, random_state=42)

# Tokenization & Embedding
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512, return_tensors="pt")
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512, return_tensors="pt")

# Encoding labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train).astype('long')  # Ensure dtype is long
y_test_encoded = label_encoder.transform(y_test).astype('long')  # Ensure dtype is long

# Dataloader
train_dataset = TensorDataset(train_encodings['input_ids'], torch.tensor(y_train_encoded))
test_dataset = TensorDataset(test_encodings['input_ids'], torch.tensor(y_test_encoded))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# CNN Model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3, 4, 5], num_filters=100):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)  # Reshape for 1D convolution
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]
        x = torch.cat(x, 1)
        x = self.fc(x)
        return x

# Define hyperparameters and their possible values
HP_EMBED_DIM = hp.HParam('embed_dim', hp.Discrete([100, 300, 500]))
HP_LR = hp.HParam('lr', hp.RealInterval(1e-5, 1e-1))
HP_NUM_FILTERS = hp.HParam('num_filters', hp.Discrete([50, 100, 150]))

METRIC_LOSS = 'loss'

# Configure TensorBoard writer and HParams
writer = SummaryWriter(log_dir='./logs/hparam_tuning')


def train_test_model(hparams):
    # Initialize model with given hyperparameters
    model = TextCNN(vocab_size=tokenizer.vocab_size, embed_dim=hparams[HP_EMBED_DIM], num_classes=len(data['Category'].unique()), num_filters=hparams[HP_NUM_FILTERS])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams[HP_LR])

    # Training loop (simplified for brevity)
    for epoch in range(5):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return loss.item()

def run(run_dir, hparams):
    loss = train_test_model(hparams)
    writer.add_hparams(
        {k.name: v for k, v in hparams.items()},  # Use the 'name' attribute of the HParam objects as keys
        {METRIC_LOSS: loss}
    )



# Train models with different hyperparameters and log results
session_num = 0
for embed_dim in HP_EMBED_DIM.domain.values:
    for lr in [1e-4, 1e-3, 1e-2]:
        for num_filters in HP_NUM_FILTERS.domain.values:
            hparams = {
                HP_EMBED_DIM: embed_dim,
                HP_LR: lr,
                HP_NUM_FILTERS: num_filters,
            }
            run_name = f"run-{session_num}"
            print(f"--- Starting trial: {run_name}")
            print({h.name: hparams[h] for h in hparams})
            run(f'./logs/hparam_tuning/{run_name}', hparams)
            session_num += 1

