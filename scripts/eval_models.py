from argparse import Namespace
import os
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
import yaml
from tqdm import tqdm

from T2M_GPT.models.modules import (
    MovementConvEncoder,
    MotionEncoderBiGRUCo
)
from T2M_GPT.options.get_eval_option import get_opt
from T2M_GPT.utils.eval_trans import (
    calculate_frechet_distance,
    calculate_activation_statistics,
)
from T2M_GPT_lightning.dataset.vq_vae_dataset import VQVAEDataset
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
        raise NotImplementedError("Inference is not implemented yet.")


def get_motion_embeddings(motions, m_lens, opt):
    movement_encoder = MovementConvEncoder(
        opt.dim_pose-4,
        opt.dim_movement_enc_hidden,
        opt.dim_movement_latent
    )
    motion_encoder = MotionEncoderBiGRUCo(
        input_size=opt.dim_movement_latent,
        hidden_size=opt.dim_motion_hidden,
        output_size=opt.dim_coemb_hidden,
        device=opt.device
    )

    checkpoint = torch.load(
        opt.checkpoints_dir,  # ???? os.path.join(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar')
        map_location=opt.device
    )
    movement_encoder.load_state_dict(checkpoint['movement_encoder'])
    motion_encoder.load_state_dict(checkpoint['motion_encoder'])

    motion_encoder.to(opt.device)
    movement_encoder.to(opt.device)

    motion_encoder.eval()
    movement_encoder.eval()

    with torch.no_grad():
        motions = motions.detach().to(opt.device).float()

        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]

        '''Movement Encoding'''
        movements = movement_encoder(motions[..., :-4]).detach()
        m_lens = m_lens // opt.unit_length
        motion_embedding = motion_encoder(movements, m_lens)
    return motion_embedding


def compute_fid(
    generated_output: VQVAEDataset,
    reference_dataset: VQVAEDataset,
    model_config: dict[str, any],
) -> float:
    """
    This function computes the FID score from the given reference and generated outputs.
    """
    motion_annotation = []
    motion_pred = []
    for i in tqdm(range(len(reference_dataset)), desc="Computing val predictions", unit="sample", leave=False):
        motion_pred.append(
            generated_output[i].reshape(-1, model_config["model_hyperparameters"]["skels_dim"])
        )
        motion_annotation.append(
            reference_dataset[i].reshape(-1, model_config["model_hyperparameters"]["skels_dim"])[: motion_pred[-1].shape[0]]
        )

    # TODO: model options
    opt_path = ''  # ???? 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    opt = get_opt(opt_path, torch.device('cuda'))
    # opt = Namespace()
    # opt_dict = vars(opt)
    # opt_dict['dim_movement_latent'] = None  # ????
    # opt_dict['dim_movement_enc_hidden'] = None  # ????
    # opt_dict['unit_length'] = None  # ????
    # opt_dict['dataset_name'] = ''  # ????
    # opt_dict['checkpoints_dir'] = ''  # ???? './checkpoints'
    # opt.device = torch.device('cuda')
    # opt.dim_pose = None  # ???? 263
    # opt.dim_motion_hidden = None  # ???? 1024
    # opt.dim_coemb_hidden = None  # ???? 512

    m_length = None # ???? len(motion)

    # Motion Prediction
    motion_pred_list = get_motion_embeddings(motion_pred, m_length, opt)
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    mu, cov= calculate_activation_statistics(motion_pred_np)

    # Motion Annotation
    motion_annotation_list = get_motion_embeddings(motion_annotation, m_length, opt)
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    return fid


def compute_apd(generated_output: VQVAEDataset, reference_dataset: VQVAEDataset) -> float:
    """
    This function computes the Average Point Distance (APD) between generated and real images.
    """
    # TODO: Implement APD computation
    raise NotImplementedError("APD computation is not implemented yet.")


def eval_models(
    models_path: dict[str, str],
    dataset_config: dict[str, any],
    model_config: dict[str, any],
) -> dict:
    # Load model
    face_model = (
        VQVAEModel.load_from_checkpoint(models_path["face_model_checkpoint"])
        if "face_model_checkpoint" in models_path
        else None
    )
    body_model = (
        VQVAEModel.load_from_checkpoint(models_path["body_model_checkpoint"])
        if "body_model_checkpoint" in models_path
        else None
    )
    hand_model = (
        VQVAEModel.load_from_checkpoint(models_path["hand_model_checkpoint"])
        if "hand_model_checkpoint" in models_path
        else None
    )
    all_model = (
        VQVAEModel.load_from_checkpoint(models_path["all_model_checkpoint"])
        if "all_model_checkpoint" in models_path
        else None
    )

    main_model = MainModel(
        face_model=face_model,
        body_model=body_model,
        hand_model=hand_model,
        all_model=all_model,
    )

    # Load reference dataset
    ref_dataset = VQVAEDataset(
        data_path=dataset_config["data_path"] if "data_path" in dataset_config else None,
        data_tensor_path=dataset_config["data_tensor_path"] if "data_tensor_path" in dataset_config else None,
        joint_size=dataset_config["joint_size"],
        window_size=dataset_config["window_size"],
        normalise=dataset_config["normalize_data"],
        is_data_has_timestamp=dataset_config["is_data_has_timestamp"],
        data_spec=dataset_config["data_spec"] if "data_spec" in dataset_config else "all",
    )

    # Generate model's output
    generated_output = main_model.inference(ref_dataset)

    # Compute evaluation metrics
    fid_score = compute_fid(generated_output, ref_dataset, model_config)
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
                models_path=experiment["models_path"],
                dataset_config=experiment["dataset_config"],
                model_config=experiment["model_config"],
            )
            f.write(f"Evaluation Output: {eval_output}\n")
        except Exception as e:
            print(f"Error occurred while evaluating models for experiment {experiment['experiment_name']}: {e}")
            f.write(f"Error occurred: {e}\n")
            continue

        f.write("\n")

    f.close()
