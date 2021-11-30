from simpletransformers.classification import ClassificationModel, ClassificationArgs
from torchtext.datasets import AG_NEWS
import pandas as pd

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

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=1, use_multiprocessing=False, use_multiprocessing_for_evaluation=False)

# Create a ClassificationModel
model = ClassificationModel(
    "bert",
    "bert-base-cased",
    use_cuda=False,
    args=model_args
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(test_df)

print(result)
