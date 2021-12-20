
local DS_READER = {
    "type": "text_classification_json",
    "tokenizer": "spacy",
    "text_key": "text",
    "label_key": "label",
    "max_sequence_length": 128
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
  
  "train_data_path": "data/agnews/train.json",
  "validation_data_path": "data/agnews/valid.json",
  "test_data_path": "data/agnews/test.json",
  
  
  "vocabulary": {"type": "from_files", "directory": "data/agnews/vocabulary.tar.gz"},
  "datasets_for_vocab_creation": ["train"],
 
  "model": {
    "type": "basic_classifier",

    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": embedding_dim,
          "trainable": true
        }
      }
    },

    "seq2seq_encoder": {
        "type": "pytorch_transformer",
        "input_dim": embedding_dim,
        "num_layers": hidden_layers,
        "feedforward_hidden_dim": embedding_dim*4,
        "num_attention_heads": 8,
        "positional_encoding": "sinusoidal",
    },
    
    "seq2vec_encoder": {
       "type": "bag_of_embeddings",
       "embedding_dim": embedding_dim,
       "averaged": true,
    }

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
      "lr": 0.00001
    }
  }
}