import optuna

def get_config_filename(arch, dataset):
    return f"training_config/{arch}_on_{dataset}.jsonnet"

def get_objective(arch, dataset):
    config_name = get_config_filename(arch, dataset)
    base_serial_dir = f"optuna/{dataset}/{arch}"
    
    def objective(trial: optuna.Trial) -> float:
        embedding_dim = trial.suggest_int("embedding_dim", 32, 1024)
        hidden_layers = trial.suggest_int("hidden_layers", 1, 6)

        executor = optuna.integration.allennlp.AllenNLPExecutor(
            trial=trial,  # trial object
            config_file=config_name,  # path to jsonnet
            serialization_dir=f"{base_serial_dir}/{trial.number}",
            metrics="best_validation_accuracy",
            include_packge="my_project"
        )
        return executor.run()
    
    def objective_cnn(trial: optuna.Trial) -> float:
        embedding_dim = trial.suggest_int("embedding_dim", 32, 1024)
        hidden_layers = trial.suggest_int("hidden_layers", 1, 6)
        kernel_size = trial.suggest_int("kernel_size", 3, 11)

        executor = optuna.integration.allennlp.AllenNLPExecutor(
            trial=trial,  # trial object
            config_file=config_name,  # path to jsonnet
            serialization_dir=f"{base_serial_dir}/{trial.number}",
            metrics="best_validation_accuracy"
        )
        return executor.run()
    
    if arch == "cnn":
        return objective_cnn
    else:
        return objective

def get_search_space(arch):
    if arch == "cnn": 
        return {
            "hidden_layers": list(range(1,7)),
            "embedding_dim": list(range(32, 1024, 128)),
            "kernel_size": [3,7,11],
        }
    else:
        return {"embedding_dim": list(range(32, 1024, 128)), "hidden_layers": list(range(1,7))}

def main():
    import sys
    arch = sys.argv[1]
    dataset = sys.argv[2]

    search_space = get_search_space(arch)
    storage = f"sqlite:///optuna/{dataset}/{arch}/trial.db"
    study = optuna.create_study(
            storage=storage,  # save results in DB
            sampler=optuna.samplers.GridSampler(search_space),
            study_name=f"optuna_{dataset}_{arch}",
            direction="maximize",
            load_if_exists=True
        )

    timeout = 60 * 60 * 10  # timeout (sec): 60*60*10 sec => 10 hours
    study.optimize(
        get_objective(arch, dataset),
        n_jobs=1,  # number of processes in parallel execution
        n_trials=200,  # number of trials to train a model
        timeout=timeout,  # threshold for executing time (sec)
    )

if __name__ == "__main__":
    main()