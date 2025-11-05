import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tqdm import tqdm

import config
from .models import MLPClassifier


class CompoundDataset(Dataset):
    def __init__(self, df):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        concatenated = np.concatenate([
            row['compound_contrast'],
            row['sentence_embedding']
        ], axis=0)
        return (
            torch.tensor(concatenated, dtype=torch.float32),
            torch.tensor(row['label'], dtype=torch.int64)
        )


def train_classifier(train_data, test_data):
    if len(train_data) == 0:
        return {'test_acc': 0.0, 'test_samples': 0}

    kfold = KFold(n_splits=config.TRAINING_CONFIG['k_folds'], shuffle=True,
                  random_state=config.TRAINING_CONFIG['random_seed'])
    train_indices = np.arange(len(train_data))

    print("K-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_indices)):
        fold_train = train_data.iloc[train_idx].reset_index(drop=True)
        fold_val = train_data.iloc[val_idx].reset_index(drop=True)

        fold_train_loader = DataLoader(
            CompoundDataset(fold_train),
            batch_size=max(1, len(fold_train) // 4),
            shuffle=True
        )
        fold_val_loader = DataLoader(
            CompoundDataset(fold_val),
            batch_size=max(1, len(fold_val))
        )

        model = MLPClassifier(
            input_dim=config.MLP_CONFIG['input_dim'],
            hidden_dims=config.MLP_CONFIG['hidden_dims'],
            dropout=config.MLP_CONFIG['dropout']
        ).to(config.DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=config.TRAINING_CONFIG['learning_rate'],
                          weight_decay=config.TRAINING_CONFIG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAINING_CONFIG['num_epochs'])

        for epoch in range(config.TRAINING_CONFIG['num_epochs']):
            model.train()
            for features, labels in fold_train_loader:
                features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

        model.eval()
        fold_preds = []
        with torch.no_grad():
            for features, _ in fold_val_loader:
                features = features.to(config.DEVICE)
                outputs = model(features)
                fold_preds.extend(outputs.argmax(dim=1).cpu().numpy())

        fold_acc = accuracy_score(fold_val['label'].values, fold_preds)
        print(f"Fold {fold + 1}: {fold_acc * 100:.2f}%")

    print("Training final model on all train data...")

    final_model = MLPClassifier(
        input_dim=config.MLP_CONFIG['input_dim'],
        hidden_dims=config.MLP_CONFIG['hidden_dims'],
        dropout=config.MLP_CONFIG['dropout']
    ).to(config.DEVICE)

    optimizer = AdamW(final_model.parameters(), lr=config.TRAINING_CONFIG['learning_rate'],
                      weight_decay=config.TRAINING_CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAINING_CONFIG['num_epochs'])

    train_loader = DataLoader(CompoundDataset(train_data), batch_size=max(1, len(train_data) // 4), shuffle=True)

    for epoch in range(config.TRAINING_CONFIG['num_epochs']):
        final_model.train()
        for features, labels in train_loader:
            features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = final_model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    final_model.eval()
    if len(test_data) > 0:
        test_loader = DataLoader(CompoundDataset(test_data), batch_size=max(1, len(test_data)))
        test_preds = []

        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(config.DEVICE)
                outputs = final_model(features)
                test_preds.extend(outputs.argmax(dim=1).cpu().numpy())

        test_acc = accuracy_score(test_data['label'], test_preds)

        return {
            'test_acc': test_acc,
            'test_samples': len(test_data),
            'model': final_model
        }

    return {'test_acc': 0.0, 'test_samples': 0}
