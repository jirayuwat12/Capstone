import os
from copy import deepcopy
from pprint import pprint

import torch
import yaml
from tqdm import tqdm

from T2M_GPT_lightning.dataset.vq_vae_dataset import MockVQVAEDataset, VQVAEDataset
from T2M_GPT_lightning.models.vqvae.vqvae import VQVAEModel

CONFIG_PATH = "./configs/eval_models.yaml"
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)
pprint(config)


class MainModel:
    def __init__(
        self,
        face_model: VQVAEModel | None = None,
        body_model: VQVAEModel | None = None,
        hand_model: VQVAEModel | None = None,
        all_model: VQVAEModel | None = None,
    ) -> None:
        """
        This class represents the main model for evaluation, encapsulating different VQVAE models.

        :param face_model: The VQVAE model for face generation.
        :param body_model: The VQVAE model for body generation.
        :param hand_model: The VQVAE model for hand generation.
        :param all_model: The VQVAE model for all generation.
        """
        is_all_model_given = all_model is not None
        is_three_model_given = face_model is not None and body_model is not None and hand_model is not None
        if not (is_all_model_given or is_three_model_given):
            raise ValueError("At least all_model or all three models (face, body, hand) must be provided.")

        self.face_model = face_model
        self.body_model = body_model
        self.hand_model = hand_model
        self.all_model = all_model

    def inference(self, reference_dataset: VQVAEDataset) -> VQVAEDataset:
        """
        This method performs inference using the main model on the provided reference dataset.

        :param reference_dataset: The dataset to perform inference on.
        :return: The dataset containing the generated outputs.
        """
        all_results = []
        if self.all_model is not None:
            for i in tqdm(range(len(reference_dataset)), desc="Computing val predictions", unit="sample", leave=False):
                # result = self.all_model(reference_dataset[i].unsqueeze(0).float().to(self.all_model.device))[0][0].detach().cpu()
                result = (reference_dataset[i].unsqueeze(0).float().to("cpu"))[0][0].detach().cpu()
                all_results.append(result)
            all_results = torch.concatenate(all_results, dim=0).to(device="cpu")

        generated_dataset = deepcopy(reference_dataset)
        generated_dataset.data = all_results

        return generated_dataset


def compute_fid(generated_output: VQVAEDataset, reference_dataset: VQVAEDataset) -> float:
    """
    This function computes the FID score from the given reference and generated outputs.
    """
    # TODO: Implement FID computation
    raise NotImplementedError("FID computation is not implemented yet.")


def compute_apd(generated_output: VQVAEDataset, reference_dataset: VQVAEDataset) -> float:
    """
    This function computes the Average Point Distance (APD) between generated and real images.
    """
    # TODO: Implement APD computation
    raise NotImplementedError("APD computation is not implemented yet.")


def eval_models(models_path: dict[str, str], dataset_config: dict[str, any]) -> dict:
    # Load model
    print(models_path)
    face_model = VQVAEModel.load_from_checkpoint(models_path["face_model"]) if "face_model" in models_path else None
    body_model = VQVAEModel.load_from_checkpoint(models_path["body_model"]) if "body_model" in models_path else None
    hand_model = (
        (VQVAEModel.load_from_checkpoint(models_path["hand_model"]) if "hand_model" in models_path else None)
        if False
        else 1
    )
    all_model = (
        (VQVAEModel.load_from_checkpoint(models_path["all_model"]) if "all_model" in models_path else None)
        if False
        else 1
    )

    main_model = MainModel(
        face_model=face_model,
        body_model=body_model,
        hand_model=hand_model,
        all_model=all_model,
    )

    # Load reference dataset
    ref_dataset = (
        VQVAEDataset(
            data_path=dataset_config["data_path"] if "data_path" in dataset_config else None,
            data_tensor_path=dataset_config["data_tensor_path"] if "data_tensor_path" in dataset_config else None,
            joint_size=dataset_config["joint_size"],
            window_size=dataset_config["window_size"],
            normalise=dataset_config["normalize_data"],
            is_data_has_timestamp=dataset_config["is_data_has_timestamp"],
            data_spec=dataset_config["data_spec"] if "data_spec" in dataset_config else "all",
        )
        if False
        else MockVQVAEDataset()
    )

    # Generate model's output
    generated_output = main_model.inference(ref_dataset)

    # Compute evaluation metrics
    fid_score = compute_fid(generated_output, ref_dataset)
    apd_score = compute_apd(generated_output, ref_dataset)

    return {
        "FID": fid_score,
        "APD": apd_score,
    }


if __name__ == "__main__":
    if not os.path.exists(config["output_file_path"]):
        os.makedirs(os.path.dirname(config["output_file_path"]), exist_ok=True)

    f = open(config["output_file_path"], "a")

    for experiment in config["experiment_set"]:
        f.write(f"Experiment: {experiment['experiment_name']}\n")

        try:
            eval_output = eval_models(
                models_path=experiment["models_path"], dataset_config=experiment["dataset_config"]
            )
            f.write(f"Evaluation Output: {eval_output}\n")
        except Exception as e:
            print(f"Error occurred while evaluating models for experiment {experiment['experiment_name']}: {e}")
            f.write(f"Error occurred: {e}\n")
            continue

        f.write("\n")

    f.close()
