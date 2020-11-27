from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from bachelorarbeit.model.classifier import SentimentClassifier
from bachelorarbeit.model.utils.analysis import accuracy_epoch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Building Sentiment Classifier starts here ----------------------------------------

def get_model(class_names) -> SentimentClassifier:
    model = SentimentClassifier(n_classes=len(class_names))
    model = model.to(device)
    return model


# Training starts here ----------------------------------------


def setup_training(train_data_loader, model: SentimentClassifier, epoch: int = 10, learning_rate: float = 2e-5,
                   correct_bias: bool = False, num_warmup_steps: int = 0) -> tuple:
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        correct_bias=correct_bias
    )
    total_steps = len(train_data_loader) * epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)
    return optimizer, total_steps, scheduler, loss_fn


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        scores = d["scores"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, scores)
        correct_predictions += torch.sum(preds == scores)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["scores"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


def train(model: SentimentClassifier, train_data_loader: DataLoader, val_data_loader: DataLoader,
          training_set: pd.DataFrame, validation_set: pd.DataFrame, epochs: int, model_name: str):
    optimizer, total_steps, scheduler, loss_fn = setup_training(
        model=model,
        train_data_loader=train_data_loader,
        epoch=epochs
    )
    history = defaultdict(list)
    best_accuracy = 0
    best_model = object()
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler,
                                            len(training_set))
        print(f'Train loss {train_loss} accuracy {train_acc}')
        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(validation_set))
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print('\n')
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            best_model = model
            best_accuracy = val_acc
    accuracy_epoch(
        history,
        model_name=model_name
    )
    return best_model
