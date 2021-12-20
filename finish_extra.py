import optuna
import pandas as pd

from allennlp.commands.train import train_model
from pathlib import Path
from allennlp.common import Params

def main():
    search_space = {
            "hidden_layers": list(range(1,7)),
            "embedding_dim": list(range(32, 1024, 128)),
            "kernel_size": [3,7,11],
    }
    study = optuna.create_study(
                storage="sqlite:///optuna/agnews/cnn/trial.db",  # save results in DB
                study_name="optuna_agnews_cnn",
                direction="maximize",
                load_if_exists=True
    )

    df = study.trials_dataframe()

    for i in range(1,7):
        for k in [3,7,11]:
            idf = df[(df["params_hidden_layers"]==i) & (df["params_kernel_size"]==k) & (df["state"]=="COMPLETE")]
            dims = sorted(set(search_space["embedding_dim"]) - set(idf["params_embedding_dim"].tolist()))
            print(i, k, dims)
            for d in dims:
                exp_dir = f"hidden_{i}__dim_{d}__kernel_{k}"
                print(exp_dir)
                params = Params.from_file("training_config/cnn_on_agnews.jsonnet", ext_vars={"hidden_layers":str(i), "embedding_dim":str(d), "kernel_size":str(k)})
                train_model(
                    params=params,
                    serialization_dir=Path("optuna/agnews/cnn")/exp_dir,
                    file_friendly_logging=False,
                    force=False,
                )


if __name__ == "__main__":
    main()