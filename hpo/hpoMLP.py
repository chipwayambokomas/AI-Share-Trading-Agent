# hpo/hpoMLP.py

import optuna
import copy
import os
import torch
import shutil

from train.trainMLP import train
from utils.utils import load_config_from_path


def _set_nested_config_value(config_dict, keys_list, value):
    d = config_dict
    for key in keys_list[:-1]:
        d = d.setdefault(key, {})
    d[keys_list[-1]] = value


def objective(trial: optuna.Trial, base_config: dict) -> float:
    trial_config = copy.deepcopy(base_config)

    for param_full_name, settings in trial_config["hpo"]["search_space"].items():
        param_type = settings.get("type", "float")
        values = settings["values"]
        if param_type == "categorical":
            param_value = trial.suggest_categorical(param_full_name, values)
        elif param_type == "int":
            param_value = trial.suggest_int(param_full_name, values[0], values[1], step=settings.get("step", 1))
        elif param_type == "float_log":
            param_value = trial.suggest_float(param_full_name, values[0], values[1], log=True)
        elif param_type == "float":
            param_value = trial.suggest_float(param_full_name, values[0], values[1])
        else:
            raise ValueError(f"Unsupported param type '{param_type}' for '{param_full_name}'")
        _set_nested_config_value(trial_config, param_full_name.split("."), param_value)

    trial_id_str = f"trial_{trial.number}"
    val_loss, temp_checkpoint_path = train(trial_config, trial_name_for_temp_checkpoint=trial_id_str)

    if temp_checkpoint_path is None or val_loss == float('inf'):
        raise optuna.exceptions.TrialPruned()

    trial.set_user_attr("temp_checkpoint_path", temp_checkpoint_path)
    trial.set_user_attr("trial_config_dict", trial_config)
    return val_loss


def hpo_completed_callback(study: optuna.Study, frozen_trial: optuna.trial.FrozenTrial):
    temp_path = frozen_trial.user_attrs.get("temp_checkpoint_path")
    project_root = study.user_attrs.get("project_root_dir", ".")

    if study.best_trial is None or frozen_trial.number != study.best_trial.number:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return

    config_best = study.best_trial.user_attrs.get("trial_config_dict")
    if config_best is None:
        return

    save_path = os.path.join(project_root, "checkpoints/MLP/MLP_best_hpo_model.pt")
    if "project_root_dir" in config_best:
        del config_best["project_root_dir"]
    _set_nested_config_value(config_best, ["training", "checkpoint_path_hpo_best"], os.path.basename(save_path))

    temp_checkpoint = study.best_trial.user_attrs.get("temp_checkpoint_path")
    if temp_checkpoint and os.path.exists(temp_checkpoint):
        checkpoint = torch.load(temp_checkpoint)
        checkpoint["config"] = config_best

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)

        try:
            os.remove(temp_checkpoint)
        except OSError:
            pass


def run_hpo(config_path: str, project_root_dir: str):
    base_config = load_config_from_path(config_path)
    base_config["project_root_dir"] = project_root_dir

    if not base_config.get("hpo", {}).get("enabled", False):
        return None

    study_name = base_config["hpo"].get("study_name", "mlp_optimization_study")
    n_trials = base_config["hpo"].get("max_trials", 20)

    study = optuna.create_study(study_name=study_name, direction="minimize")
    study.set_user_attr("project_root_dir", project_root_dir)

    study.optimize(lambda trial: objective(trial, base_config), n_trials=n_trials, callbacks=[hpo_completed_callback])

    best_params = None
    if study.best_trial:
        best_params = study.best_trial.params

    temp_dir = os.path.join(project_root_dir, "checkpoints/MLP/temp_trial_models")
    if base_config["hpo"].get("cleanup_temp_trial_dir_after_hpo", True) and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            pass

    return best_params
