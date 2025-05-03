# Copyright (c) Hao Meng. All Rights Reserved.

import logging
import os
import time

import torch
from tqdm import tqdm


def normalize(vec):
    eps = 1e-8
    eps = 1e-8
    len = torch.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / (len + eps)
    return vec / (len + eps)


def calculate_interval_loss(xmin: torch.Tensor, xmax: torch.Tensor, X: torch.Tensor):
    # print("calculate_interval_loss", xmin.shape, xmax.shape, X.shape)

    interval_loss = 0
    # X (B ,20 ,N)
    # X[: , i] =
    # xmin (B , 20)
    for i in range(X.shape[1]):  # 20 or 4
        X_outlier = X[:, i][(X[:, i] < xmin[:, i]) | (X[:, i] > xmax[:, i])]
        less_min = xmin[:, i, torch.newaxis] - X_outlier
        more_max = X_outlier - xmax[:, i]
        div = 1
        if len(X_outlier.shape) >= 3:
            div = X_outlier.shape[2]  #
        interval_loss += torch.max(less_min, more_max).clamp(0).sum() / div

    # div = max(X.shape[2], 1)
    return interval_loss / X.shape[1]


# joint Sequence*126*3
def calculate_bmc_loss(origin_joints, reconstructed_joints, batch_size=1):
    # print(origin_joints.shape, reconstructed_joints.shape)

    origin_joints = origin_joints.reshape(batch_size, -1, 21, 6)
    reconstructed_joints = reconstructed_joints.reshape(batch_size, -1, 21, 6)
    xhat_rh_joints = reconstructed_joints[:, :, :, 0:3]
    xhat_lh_joints = reconstructed_joints[:, :, :, 3:6]
    rh_joints = origin_joints[:, :, :, 0:3]
    lh_joints = origin_joints[:, :, :, 3:6]
    rh_loss = bmc_loss(rh_joints, xhat_rh_joints)
    lh_loss = bmc_loss(lh_joints, xhat_lh_joints)
    return rh_loss + lh_loss


def bmc_loss(x_joints, xhat_joints):
    # joint shape = (B , N, 21, 3)
    # all_start = time.time()
    snap_child = torch.arange(1, 21)
    snap_parents = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]

    x_kin_chain = x_joints[:, :, snap_child] - x_joints[:, :, snap_parents]
    x_kin_chain = x_joints[:, :, snap_child] - x_joints[:, :, snap_parents]

    xhat_kin_chain = xhat_joints[:, :, snap_child] - xhat_joints[:, :, snap_parents]
    xhat_kin_chain = xhat_joints[:, :, snap_child] - xhat_joints[:, :, snap_parents]

    # print("XKIN_CHAIN", x_kin_chain.shape) #(B , N , 20 , 3)

    x_bone_lens = torch.linalg.norm(x_kin_chain, ord=2, axis=-1)  # (B*N*20)
    x_bone_lens = torch.linalg.norm(x_kin_chain, ord=2, axis=-1)  # (B*N*20)

    # print("X_BONE_LENS", x_bone_lens.shape)

    max_bone_len = (torch.max(x_bone_lens, dim=1).values).unsqueeze(-1)  # (B*20,)
    min_bone_len = (torch.min(x_bone_lens, dim=1).values).unsqueeze(-1)  # (B*20,)

    # xhat_kin_chain = torch.stack(xhat_kin_chain, dim=1)  # (N, 20, 3)
    xhat_kin_chain2 = xhat_kin_chain.swapaxes(2, 1)  # (B*20*N*3)
    xhat_bone_lens = torch.linalg.norm(xhat_kin_chain2, ord=2, axis=-1)  # (B*20*N)
    xhat_kin_chain2 = xhat_kin_chain.swapaxes(2, 1)  # (B*20*N*3)
    xhat_bone_lens = torch.linalg.norm(xhat_kin_chain2, ord=2, axis=-1)  # (B*20*N)

    # print("XHAT_BONE_LENS", xhat_bone_lens.shape)
    # print("SHAPE BEF FUNC",min_bone_len.shape, max_bone_len.shape, xhat_bone_lens.shape)
    # f_start = time.time()
    bl_loss = calculate_interval_loss(min_bone_len, max_bone_len, xhat_bone_lens)
    # f_end = time.time()
    # elapsed = f_end - f_start
    # print(f"BL calculate_bmc_loss execution time: {elapsed:.4f} seconds")

    # calculate root bone limits
    root_bones = x_kin_chain[:, :, [0, 5, 9, 13, 17], :]
    hat_root_bones = xhat_kin_chain[:, :, [0, 5, 9, 13, 17], :]

    eps = 1e-2
    normals = normalize(torch.linalg.cross(root_bones[:, :, 1:], root_bones[:, :, :-1]))
    # print("NORMALS", normals.shape) #(N, 4, 3)
    edge_normals = torch.zeros_like(root_bones, device=x_joints.device)
    edge_normals[:, :, 0] = normals[:, :, 0]
    edge_normals[:, :, 4] = normals[:, :, 3]
    edge_normals[:, :, 1:4] = normalize(normals[:, :, 1:] + normals[:, :, :-1])

    # x_curvatures = torch.zeros([x_joints.shape[0],x_joints.shape[1], 4], device=x_joints.device)
    # x_PHI = torch.zeros([x_joints.shape[0],x_joints.shape[1], 4], device=x_joints.device)

    id_first = [1, 2, 3, 4]
    id_second = [0, 1, 2, 3]
    e_tmp = edge_normals[:, :, id_first] - edge_normals[:, :, id_second]
    b_tmp = root_bones[:, :, id_first] - root_bones[:, :, id_second]  # B*N*4*3
    b_tmp_norm = torch.linalg.norm(b_tmp, ord=2, axis=-1)  # B*N*4

    x_curvatures = torch.sum(e_tmp * b_tmp, axis=-1) / (b_tmp_norm**2 + eps)  # B*N*4
    x_PHI = torch.sum(root_bones[:, :, id_second] * root_bones[:, :, id_first], axis=-1)

    tmp1 = torch.linalg.norm(root_bones[:, :, id_second], ord=2, axis=-1)
    tmp2 = torch.linalg.norm(root_bones[:, :, id_first], ord=2, axis=-1)
    x_PHI /= tmp1 * tmp2 + eps
    x_PHI = torch.acos(x_PHI)

    normals = normalize(torch.linalg.cross(hat_root_bones[:, :, 1:], hat_root_bones[:, :, :-1]))
    # print("NORMALS", normals.shape) #(N, 4, 3)
    edge_normals = torch.zeros_like(hat_root_bones, device=x_joints.device)
    edge_normals[:, :, 0] = normals[:, :, 0]
    edge_normals[:, :, 4] = normals[:, :, 3]
    edge_normals[:, :, 1:4] = normalize(normals[:, :, 1:] + normals[:, :, :-1])

    # xhat_PHI_t = torch.zeros([x_joints.shape[0],x_joints.shape[1], 4], device=x_joints.device )
    # xhat_curvatures_t = torch.zeros([x_joints.shape[0],x_joints.shape[1], 4], device=x_joints.device )

    e_tmp = edge_normals[:, :, id_first] - edge_normals[:, :, id_second]
    b_tmp = hat_root_bones[:, :, id_first] - hat_root_bones[:, :, id_second]
    b_tmp_norm = torch.linalg.norm(b_tmp, ord=2, axis=-1)  # N

    xhat_curvatures_t = torch.sum(e_tmp * b_tmp, axis=-1) / (b_tmp_norm**2 + eps)
    xhat_PHI_t = torch.sum(hat_root_bones[:, :, id_second] * hat_root_bones[:, :, id_first], axis=-1)

    tmp1 = torch.linalg.norm(hat_root_bones[:, :, id_second], ord=2, axis=-1)
    tmp2 = torch.linalg.norm(hat_root_bones[:, :, id_first], ord=2, axis=-1)
    xhat_PHI_t /= tmp1 * tmp2 + eps
    xhat_PHI_t = torch.acos(xhat_PHI_t)

    max_curvatures = (torch.max(x_curvatures, dim=1).values).unsqueeze(-1)  # (B,4,1)
    min_curvatures = (torch.min(x_curvatures, dim=1).values).unsqueeze(-1)  # (B,4,1)
    xhat_curvatures_t = xhat_curvatures_t.swapaxes(2, 1)  # (B*4*N)

    # print("BEF CURVE FUNC", min_curvatures.shape, max_curvatures.shape, xhat_curvatures.shape)
    ##############
    # f_start = time.time()
    curvature_loss = calculate_interval_loss(min_curvatures, max_curvatures, xhat_curvatures_t)
    # f_end = time.time()
    # elapsed = f_end - f_start
    # print(f"CURVE calculate_bmc_loss execution time: {elapsed:.4f} seconds")

    max_PHI = (torch.max(x_PHI, dim=1).values).unsqueeze(-1)  # (B,4,1)
    min_PHI = (torch.min(x_PHI, dim=1).values).unsqueeze(-1)  # (B,4,1)
    xhat_PHI_t = xhat_PHI_t.swapaxes(2, 1)  # (B,4*N)

    # print("BEF PHI FUNC", min_PHI.shape, max_PHI.shape, xhat_PHI.shape)
    ##############

    # f_start = time.time()
    phi_loss = calculate_interval_loss(min_PHI, max_PHI, xhat_PHI_t)
    # f_end = time.time()
    # elapsed = f_end - f_start
    # print(f"PHI calculate_bmc_loss execution time: {elapsed:.4f} seconds")

    # all_end = time.time()
    # print(f"ALL bmc_loss execution time: {all_end - all_start:.4f} seconds")
    if torch.isnan(bl_loss) or torch.isnan(curvature_loss) or torch.isnan(phi_loss):
        import datetime

        timestamp = datetime.datetime.now().isoformat()
        with open("loss_nan_log.txt", "w") as f:
            f.write(
                f"{timestamp} - bl_loss: {bl_loss.item()}, curvature_loss: {curvature_loss.item()}, phi_loss: {phi_loss.item()}\n"
            )
            f.write(
                f"min_bone_len: {min_bone_len.shape}, max_bone_len: {max_bone_len.shape} ,xhat_bone_lens: {xhat_bone_lens.shape} \n"
            )
            f.write(
                f"min_curvatures: {min_curvatures.shape}, max_curvatures: {max_curvatures.shape} ,xhat_curvatures: {xhat_curvatures_t.shape} \n"
            )
            f.write(f"min_PHI: {min_PHI.shape}, max_PHI: {max_PHI.shape} ,xhat_PHI: {xhat_PHI_t.shape} \n\n")

    # print("BL", bl_loss, "CURVATURE", curvature_loss, "PHI", phi_loss)
    return 0.1 * (bl_loss + curvature_loss + phi_loss) / x_joints.shape[0]
