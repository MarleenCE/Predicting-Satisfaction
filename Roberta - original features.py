#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import RobertaConfig
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import TensorDataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch
import time
import nltk
from nltk.tokenize import word_tokenize
import random
from tqdm import tqdm
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

train_df = pd.read_csv("C:\\Users\\Mathijs\\Documents\\marleen\\train4.csv", header = 0, sep= ',')
test_df = pd.read_csv("C:\\Users\\Mathijs\\Documents\\marleen\\test4.csv", header = 0, sep= ',')

train_df.drop_duplicates(inplace=True)

y = train_df['Grade']
train_df, validation_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=y)  
print("Train set shape: ", train_df.shape)
print("Test set shape: ", test_df.shape)
print("Validation set shape: ", validation_df.shape)


# Define CustomSequenceClassification class
class CustomSequenceClassification(nn.Module):
    def __init__(self, num_labels=1, pretrained_model_name="pdelobelle/robbert-v2-dutch-base", num_numeric_features=3,  dropout_rate=0.2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        
        
        # Adding a dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Regression head (modify as needed)
        self.regression_head = nn.Linear(self.roberta.config.hidden_size + num_numeric_features, num_labels)
        
    def forward(self, input_ids, attention_mask, travel_month, travel_duration, land):
        # RoBERTa model forward pass
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Combine RoBERTa output with numerical features
        combined_features = torch.cat((pooled_output, travel_month, travel_duration, land), dim=1)
        
        # Regression head
        logits = self.regression_head(combined_features)
        
        return logits
    

# Preparing 
tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")   

def prepare_data(data_df):
    data_df = data_df.dropna(subset=['Gespreksdata'])
    data_df['Gespreksdata'] = data_df['Gespreksdata'].astype(str)
    inputs_text = tokenizer(data_df["Gespreksdata"].tolist(), padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    travel_month = torch.tensor(data_df["Travel Month"].values, dtype=torch.float).unsqueeze(1)  
    travel_duration = torch.tensor(data_df["Travel Duration"].values, dtype=torch.float).unsqueeze(1)  
    land = torch.tensor(data_df["Land"].values, dtype=torch.float).unsqueeze(1)  
    
    # Convert labels to tensor
    labels = torch.tensor(data_df["Grade"].values, dtype=torch.float)  
    
    return TensorDataset(inputs_text.input_ids, inputs_text.attention_mask, travel_month, travel_duration, land, labels) 
 

# Prepare train and test datasets
train_dataset = prepare_data(train_df)
validation_dataset = prepare_data(validation_df)
test_dataset = prepare_data(test_df)

# Step 3: Training Setup
batch_size = 32
num_epochs = 20
learning_rate = 1e-5

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# initialize model
config = RobertaConfig.from_pretrained("pdelobelle/robbert-v2-dutch-base")
num_extra_dims = 3  # Adjust based on the number of additional numerical features
model = CustomSequenceClassification(num_labels=1, pretrained_model_name="pdelobelle/robbert-v2-dutch-base", num_numeric_features=num_extra_dims)
            
# freezing first x layers
for name, param in model.named_parameters():
    if name.startswith('roberta.encoder.layer') and int(name.split('.')[3]) < 8:
        param.requires_grad = False
        

#freeze all layers
#for param in model.roberta.parameters():
#        param.requires_grad = False


########
def calculate_weights(labels):
    # You can adjust the weights and boundary as needed
    weights = torch.where(labels <= 5.5, 1.5, 1.0)
    return weights
########

# Initialize optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss(reduction='none')


print("batch size: ", batch_size)
print("learning rate: ", learning_rate)

#########
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        """
        :param patience: Number of epochs to wait after min has been hit. After this number of epochs, training stops.
        :param min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.patience_count = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.patience_count = 0
        else:
            self.patience_count += 1
            if self.patience_count >= self.patience:
                self.early_stop = True

# Usage within your training loop:
early_stopper = EarlyStopping(patience=3, min_delta=0.001)
########

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)


# Verify layer freezing
for i, layer in enumerate(model.roberta.encoder.layer):
    params_frozen = all(not p.requires_grad for p in layer.parameters())
    print(f"Layer {i} frozen: {params_frozen}")

train_losses = []  # List to store training losses
val_losses = []    # List to store validation losses

# Define the directory and file path for saving the model
directory = 'D:\\tempmarl'
best_model_path = os.path.join(directory, 'best_model_original.pth')

# Ensure the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory created: {directory}")
else:
    print(f"Directory already exists: {directory}")

# Initialize variables
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        input_ids, attention_mask, travel_month, travel_duration, land, labels = [t.to(device) for t in batch] 
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask, travel_month, travel_duration, land)
        logits = outputs.squeeze(1)
        losses = criterion(logits, labels)
        weights = calculate_weights(labels).to(device)  # Ensure weights are on the same device
        weighted_losses = losses * weights
        loss = weighted_losses.mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    # Calculate average training loss for the epoch
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in validation_loader:  
            input_ids, attention_mask, travel_month, travel_duration, land, labels = [t.to(device) for t in batch]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, travel_month=travel_month, travel_duration=travel_duration, land=land)
            
            logits = outputs.squeeze(1)
            losses = criterion(logits, labels)
            weights = calculate_weights(labels).to(device)
            weighted_losses = losses * weights
            loss = weighted_losses.mean()
            total_val_loss += loss.item()
    
    # Calculate average validation loss for the epoch
    avg_val_loss = total_val_loss / len(validation_loader)
    val_losses.append(avg_val_loss)
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)  # Save best model weights
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    
    # Call to EarlyStopping
    early_stopper(avg_val_loss)
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break


end_time = time.time()
elapsed_time = end_time - start_time
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Execution time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")



# Plot training and validation losses
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

# Step 5: Evaluation ON TEST
model.eval()
with torch.no_grad():
    # Prepare test data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    total_predictions = []
    total_labels = []
    for batch in test_loader:
        input_ids, attention_mask, travel_month, travel_duration, land, labels = [t.to(device) for t in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, travel_month=travel_month, travel_duration=travel_duration, land=land)
        predictions = outputs.squeeze(1)
        total_predictions.extend(predictions.cpu().numpy())
        total_labels.extend(labels.cpu().numpy())

# Calculate Mean Squared Error (MSE) and MAE
mae = mean_absolute_error(total_labels, total_predictions)
mse = mean_squared_error(total_labels, total_predictions)
print("Mean Absolute Error (MAE) Test:", mae)
print("Mean Squared Error (MSE) Test:", mse)

visualization_data = np.column_stack((total_labels, total_predictions))

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(visualization_data[:, 0], visualization_data[:, 1], color='brown', alpha=0.5)
plt.plot([1, 10], [1, 10], color='black', linestyle='--')  # Diagonal line for reference
plt.title('Actual vs Predicted Grades on Test set')
plt.xlabel('Actual Grade')
plt.ylabel('Predicted Grade')
plt.xlim(1, 10)
plt.ylim(1, 10)
plt.grid(True)
plt.show()

grades = np.unique(visualization_data[:, 0])
plt.figure(figsize=(10, 6))
for grade in grades:
    grade_data = visualization_data[visualization_data[:, 0] == grade][:, 1]
    plt.boxplot(grade_data, positions=[grade], widths=0.5, patch_artist=True, boxprops=dict(facecolor='brown'))
plt.plot([1, 10], [1, 10], color='black', linestyle='--')  # Diagonal line for reference
plt.title('Actual vs Predicted Grades on Test set')
plt.xlabel('Grade')
plt.ylabel('Predicted Grade')
plt.xlim(0.5, 10.5)
plt.ylim(1, 10)
plt.grid(True)
plt.show()

