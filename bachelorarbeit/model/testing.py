import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from bachelorarbeit.model.training import eval_model, device
from bachelorarbeit.model.utils.analysis import show_confusion_matrix
from bachelorarbeit.model.classifier import SentimentClassifier

RANDOM_SEED = 42


def run_test(model, df_test, test_data_loader):
    loss_fn = nn.CrossEntropyLoss().to(device)
    test_acc, _ = eval_model(model, test_data_loader, loss_fn, device, len(df_test))
    return test_acc.item()


def get_predictions(best_model: SentimentClassifier, data_loader):
    best_model = best_model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["input_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["scores"].to(device)

            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values


def test(df_test: pd.DataFrame, test_data_loader: DataLoader, best_model: SentimentClassifier, class_names: list, model_name:str) -> None:
    test_acc = run_test(best_model, df_test, test_data_loader)
    print("The accuracy on the test data: ", test_acc)
    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(best_model, test_data_loader)
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    show_confusion_matrix(df_cm, model_name=model_name)
    return None
