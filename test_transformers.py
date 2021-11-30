import pandas as pd
from transformers import BertTokenizer
from Classifiers import BertClassifier
from Utils import train, evaluate
from torchtext.datasets import AG_NEWS

train_iter, test_iter = AG_NEWS(root='.data',split=('train','test'))

train_data = []
for sample in train_iter:
    train_data.append([sample[1],sample[0]])

train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

test_data = []
for sample in test_iter:
    test_data.append([sample[1],sample[0]])

test_df = pd.DataFrame(test_data)
test_df.columns = ["text", "labels"]

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
             
EPOCHS = 5
model = BertClassifier(num_classes=4)
LR = 1e-6
              
train(model, train_df, LR, EPOCHS, tokenizer)

evaluate(model, test_df, tokenizer)