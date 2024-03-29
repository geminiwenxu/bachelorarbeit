from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from pkg_resources import resource_filename
from torch import nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from bachelorarbeit.model import logger
from bachelorarbeit.model.classifier import SentimentClassifier
from bachelorarbeit.model.utils.analysis import plot_training_results


# Training starts here ----------------------------------------

def setup_training(train_data_loader, model: SentimentClassifier, device: torch.device, model_name, epoch: int,
                   learning_rate: float, correct_bias: bool, num_warmup_steps: int) -> tuple:
    logger.info(
        f"{model_name} --> Setting up training procedure with parameters: epoch: {epoch}, learning_rate: {learning_rate}, correct_bias: {correct_bias}, num_warmup_steps: {num_warmup_steps}")
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
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)
        _, preds = torch.max(outputs,
                             dim=1)
        loss = loss_fn(outputs, scores)
        correct_predictions += torch.sum(preds == scores)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),
                                 max_norm=1.0)
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
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            _, preds = torch.max(outputs,
                                 dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


def train(model: SentimentClassifier, device: torch.device, train_data_loader: DataLoader, val_data_loader: DataLoader,
          training_set: pd.DataFrame, validation_set: pd.DataFrame, epochs: int, model_name: str, correct_bias: bool,
          learning_rate: float, num_warmup_steps: int):
    optimizer, total_steps, scheduler, loss_fn = setup_training(
        model=model,
        train_data_loader=train_data_loader,
        epoch=epochs,
        device=device,
        model_name=model_name,
        correct_bias=correct_bias,
        learning_rate=learning_rate,
        num_warmup_steps=num_warmup_steps
    )
    logger.info(f"{model_name} --> Starting Training Procedure:")
    history = defaultdict(list)
    best_accuracy = 0
    best_model = object()
    for epoch in range(epochs):
        epoch_timer = datetime.now()
        start_timer = datetime.now()
        logger.info(f"{model_name} --> Epoch {epoch + 1}/{epochs}")
        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler,
                                            len(training_set))
        logger.info(f"{model_name} --> Training loss: {train_loss} --- accuracy: {train_acc}")
        logger.info(f"{model_name} --> Time to complete training: {datetime.now() - start_timer}")
        start_timer = datetime.now()
        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(validation_set))
        logger.info(f"{model_name} --> Validation loss: {val_loss} --- accuracy: {val_acc}")
        logger.info(f"{model_name} --> Time to complete validation: {datetime.now() - start_timer}")
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)

        if val_acc > best_accuracy:
            best_model = model
            best_accuracy = val_acc
        logger.info(
            f"{model_name} --> Time to complete training & validation of epoch {epoch}: {datetime.now() - epoch_timer}")
    logger.info(f"{model_name} --> Training Procedure Complete!")
    plot_training_results(
        history,
        model_name=model_name
    )
    filename = resource_filename(__name__, f'../../models/{model_name}_model_opt.pth')
    logger.info(f"{model_name} --> Saving optimal model to disk: {filename}")
    torch.save(model, filename)
    return best_model
