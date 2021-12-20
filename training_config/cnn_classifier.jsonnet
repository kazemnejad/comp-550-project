
local DS_READER = {
    "type": "text_classification_json",
    "tokenizer": "spacy",
    "text_key": "text",
    "label_key": "label"
};

local seed = 42;

local embedding_dim = std.parseInt(std.extVar('embedding_dim'));
local hidden_layers = std.parseInt(std.extVar('hidden_layers'));

{
  "numpy_seed": seed,
  "pytorch_seed": seed,
  "random_seed": seed,
 
  "dataset_reader": DS_READER,
  "validation_dataset_reader": DS_READER,
  
  "train_data_path": "data/ag_news_train.json",
  "validation_data_path": "data/ag_news_valid.json",
  "test_data_path": "data/ag_news_test.json",
  
  
  "vocabulary": {"type": "from_files", "directory": "vocabulary.tar.gz"},
  // "vocabulary": {
  //      "max_vocab_size": 20000,
  //      "tokens_to_add": {
  //        "tokens": ["@@@CLS@@@"],
  //      },
  // },
  "datasets_for_vocab_creation": ["train"],
 
  "model": {
    "type": "lyr_cnn_classifier",

    "embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": embedding_dim,
          "trainable": true
        }
      }
    },

    "embedding_dim": embedding_dim,
    "hidden_dim": embedding_dim,
    "num_layers": hidden_layers,
    "kernel_size": 4,
  },
  "data_loader": {
    "batch_size": 256,
    "shuffle": true,
    "num_workers": 4,
    "max_instances_in_memory": 1024,
  },
 
  "evaluate_on_test": true, 
 
  "trainer": {
    "num_epochs": 10,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
      "lr": 0.0001
    }
  }
}