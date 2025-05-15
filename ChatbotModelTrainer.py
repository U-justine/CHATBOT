import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import logging
import os
from typing import List, Tuple, Dict
import json


class ChatbotDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ChatbotTrainer:
    def __init__(self, model_name: str = 'bert-base-uncased', num_labels: int = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.label2id = None
        self.id2label = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def prepare_data(self, data_path: str) -> Tuple[Dataset, Dataset]:
        """
        Prepare training and validation datasets from Excel file
        """
        # Load data
        df = pd.read_excel(data_path)

        # Create label mappings
        unique_answers = df['Answer'].unique()
        self.label2id = {label: idx for idx, label in enumerate(unique_answers)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.num_labels = len(unique_answers)

        # Convert answers to label indices
        labels = df['Answer'].map(self.label2id).values

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['Question'].values,
            labels,
            test_size=0.2,
            random_state=42
        )

        # Create datasets
        train_dataset = ChatbotDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = ChatbotDataset(val_texts, val_labels, self.tokenizer)

        return train_dataset, val_dataset

    def initialize_model(self):
        """
        Initialize the model with the correct number of labels
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.model.to(self.device)

    def train(self,
              train_dataset: Dataset,
              val_dataset: Dataset,
              batch_size: int = 16,
              num_epochs: int = 5,
              learning_rate: float = 2e-5,
              save_path: str = 'chatbot_model'):
        """
        Train the model
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            train_steps = 0

            train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

            for batch in train_progress:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.model.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_train_loss += loss.item()
                train_steps += 1

                loss.backward()
                optimizer.step()

                train_progress.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})

            avg_train_loss = total_train_loss / train_steps

            # Validation
            self.model.eval()
            total_val_loss = 0
            val_steps = 0

            for batch in tqdm(val_loader, desc='Validation'):
                with torch.no_grad():
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = outputs.loss
                    total_val_loss += loss.item()
                    val_steps += 1

            avg_val_loss = total_val_loss / val_steps

            self.logger.info(f'Epoch {epoch + 1}:')
            self.logger.info(f'Average training loss: {avg_train_loss:.3f}')
            self.logger.info(f'Average validation loss: {avg_val_loss:.3f}')

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)

                # Save label mappings
                with open(os.path.join(save_path, 'label_mappings.json'), 'w') as f:
                    json.dump({
                        'label2id': self.label2id,
                        'id2label': self.id2label
                    }, f)

    def predict(self, text: str) -> str:
        """
        Make a prediction for a single text input
        """
        self.model.eval()

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()

        return self.id2label[predicted_label]


def main():
    # Initialize trainer
    trainer = ChatbotTrainer()

    # Prepare data
    train_dataset, val_dataset = trainer.prepare_data(r'C:\Users\HP\OneDrive\Desktop\ICT_QA.xlsx')

    # Initialize model
    trainer.initialize_model()

    # Train model
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=16,
        num_epochs=5,
        learning_rate=2e-5,
        save_path='chatbot_model'
    )


if __name__ == "__main__":
    main()
