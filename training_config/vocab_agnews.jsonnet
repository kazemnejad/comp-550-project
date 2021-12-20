
local DS_READER = {
    "type": "text_classification_json",
    "tokenizer": "spacy",
    "text_key": "text",
    "label_key": "label"
};

local seed = 42;

{
  "numpy_seed": seed,
  "pytorch_seed": seed,
  "random_seed": seed,
 
  "dataset_reader": DS_READER,
  "validation_dataset_reader": DS_READER,
  
  "train_data_path": "data/ag_news/train.json",
  
  "vocabulary": {
       "max_vocab_size": 20000,
  },

  "data_loader": {
    "batch_size": 256,
    "shuffle": false,
    "num_workers": 4,
    "max_instances_in_memory": 1024,
  },
}