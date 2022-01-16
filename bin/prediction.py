import pandas as pd
import torch
import yaml
from pkg_resources import resource_filename
from sklearn.metrics import classification_report
from transformers import BertTokenizer

from bachelorarbeit.model.classifier import SentimentClassifier


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


if __name__ == "__main__":
    path = '/Users/geminiwenxu/PycharmProjects/bachelorarbeit/models/multi_all_model_opt.pth'
    config = get_config('/../config/config.yaml')
    class_names = config['class_names']
    dropout_ratio = config['dropout_ratio']
    device = torch.device("cpu")
    df = pd.read_csv('/Users/geminiwenxu/PycharmProjects/bachelorarbeit/text_report/test_validation.csv', sep=';',
                     header=None, index_col=False,
                     names=['gold_label', 'text', 'lang'])
    # names=['index', 'num', 'actual', 'prediction', 'probability', 'text', 'source'])
    print(df)
    model = SentimentClassifier(len(class_names), dropout_ratio)
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage).module.state_dict())
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    predict = []
    for index, row in df.iterrows():
        print(index)
        encoded_review = tokenizer.encode_plus(row['text'],
                                               add_special_tokens=True,
                                               max_length=160,
                                               return_token_type_ids=False,
                                               padding='max_length',
                                               truncation=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')
        input_ids = encoded_review['input_ids'].to(device)

        attention_mask = encoded_review['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        predict.append(prediction)

    report = classification_report(df.actual.to_list(), predict, target_names=class_names)
    print(report)
