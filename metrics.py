from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(preds, labels, dataname='weibo'):
    acc = accuracy_score(labels, preds)
    if dataname == 'weibo':
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')
    else:
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
    
    macro_f1 = f1_score(labels, preds, average='macro')
    return acc, precision, recall, f1, macro_f1