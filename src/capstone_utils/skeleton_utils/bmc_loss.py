# Copyright (c) Hao Meng. All Rights Reserved.

import os

import torch
from tqdm import tqdm


def normalize(vec):
    len = torch.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / len


def calculate_interval_loss(xmin: torch.Tensor, xmax: torch.Tensor, X: torch.Tensor):
    # print("calculate_interval_loss", xmin.shape, xmax.shape, X.shape)
    interval_loss = 0
    # X shape = (20, N)
    # xmin shape = (20,)
    # xmax shape = (20,)
    for i in range(X.shape[0]):
        X_outlier = X[i][(X[i] < xmin[i]) | (X[i] > xmax[i])]
        less_min = xmin[i, torch.newaxis] - X_outlier
        more_max = X_outlier - xmax[i]
        interval_loss += torch.max(less_min, more_max).clamp(0).sum()

    return interval_loss / X.shape[1] * X.shape[0]


# joint Sequence*126*3
def calculate_bmc_loss(origin_joints, reconstructed_joints, batch_size=1):
    origin_joints = origin_joints.reshape(-1, 21, 6)
    reconstructed_joints = reconstructed_joints.reshape(-1, 21, 6)
    xhat_rh_joints = reconstructed_joints[:, :, 0:3]
    xhat_lh_joints = reconstructed_joints[:, :, 3:6]
    rh_joints = origin_joints[:, :, 0:3]
    lh_joints = origin_joints[:, :, 3:6]
    rh_loss = bmc_loss(rh_joints, xhat_rh_joints)
    lh_loss = bmc_loss(lh_joints, xhat_lh_joints)
    return rh_loss + lh_loss


def bmc_loss(x_joints, xhat_joints):
    # joint shape = (N, 21, 3)
    snap_child = torch.arange(1, 21)
    snap_parents = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]

    x_kin_chain = x_joints[:, snap_child] - x_joints[:, snap_parents]

    xhat_kin_chain = xhat_joints[:, snap_child] - xhat_joints[:, snap_parents]

    # print("XKIN_CHAIN", x_kin_chain.shape) #(N , 20 , 3)

    x_bone_lens = torch.linalg.norm(x_kin_chain, ord=2, axis=-1)  # (N*20)

    # print("X_BONE_LENS", x_bone_lens.shape)

    max_bone_len = torch.min(x_bone_lens, dim=0).values  # (20,)
    min_bone_len = torch.max(x_bone_lens, dim=0).values  # (20,)

    # xhat_kin_chain = torch.stack(xhat_kin_chain, dim=1)  # (N, 20, 3)
    xhat_kin_chain = xhat_kin_chain.swapaxes(1, 0)  # (20*N*3)
    xhat_bone_lens = torch.linalg.norm(xhat_kin_chain, ord=2, axis=-1)  # (20*N)

    # print("XHAT_BONE_LENS", xhat_bone_lens.shape)

    bl_loss = calculate_interval_loss(min_bone_len, max_bone_len, xhat_bone_lens)
    # calculate root bone limits
    root_bones = x_kin_chain[:, [0, 5, 9, 13, 17], :]
    normals = normalize(torch.linalg.cross(root_bones[:, 1:], root_bones[:, :-1]))
    # print("NORMALS", normals.shape) #(N, 4, 3)
    edge_normals = torch.zeros_like(root_bones)
    edge_normals[:, 0] = normals[:, 0]
    edge_normals[:, 4] = normals[:, 3]
    edge_normals[:, 1:4] = normalize(normals[:, 1:] + normals[:, :-1])

    x_curvatures = torch.zeros([x_joints.shape[0], 4], device=x_joints.device)
    x_PHI = torch.zeros([x_joints.shape[0], 4], device=x_joints.device)
    for i in range(4):
        e_tmp = edge_normals[:, i + 1] - edge_normals[:, i]
        b_tmp = root_bones[:, i + 1] - root_bones[:, i]
        b_tmp_norm = torch.linalg.norm(b_tmp, ord=2, axis=-1)  # N

        x_curvatures[:, i] = torch.sum(e_tmp * b_tmp, axis=-1) / (b_tmp_norm**2)
        x_PHI[:, i] = torch.sum(root_bones[:, i] * root_bones[:, i + 1], axis=-1)

        tmp1 = torch.linalg.norm(root_bones[:, i], ord=2, axis=-1)
        tmp2 = torch.linalg.norm(root_bones[:, i + 1], ord=2, axis=-1)
        x_PHI[:, i] /= tmp1 * tmp2
        x_PHI[:, i] = torch.arccos(x_PHI[:, i])

    xhat_curvatures = torch.zeros([x_joints.shape[0], 4], device=x_joints.device)
    xhat_PHI = torch.zeros([x_joints.shape[0], 4], device=x_joints.device)
    for i in range(4):
        e_tmp = edge_normals[:, i + 1] - edge_normals[:, i]
        b_tmp = root_bones[:, i + 1] - root_bones[:, i]
        b_tmp_norm = torch.linalg.norm(b_tmp, ord=2, axis=-1)  # N

        xhat_curvatures[:, i] = torch.sum(e_tmp * b_tmp, axis=-1) / (b_tmp_norm**2)
        xhat_PHI[:, i] = torch.sum(root_bones[:, i] * root_bones[:, i + 1], axis=-1)

        tmp1 = torch.linalg.norm(root_bones[:, i], ord=2, axis=-1)
        tmp2 = torch.linalg.norm(root_bones[:, i + 1], ord=2, axis=-1)
        xhat_PHI[:, i] /= tmp1 * tmp2
        xhat_PHI[:, i] = torch.arccos(xhat_PHI[:, i])

    max_curvatures = torch.max(x_curvatures, dim=0).values  # (4,)
    min_curvatures = torch.min(x_curvatures, dim=0).values  # (4,)
    xhat_curvatures = xhat_curvatures.swapaxes(1, 0)  # (4*N)
    curvature_loss = calculate_interval_loss(min_curvatures, max_curvatures, xhat_curvatures)

    max_PHI = torch.max(x_PHI, dim=0).values  # (4,)
    min_PHI = torch.min(x_PHI, dim=0).values  # (4,)
    xhat_PHI = xhat_PHI.swapaxes(1, 0)  # (4*N)
    phi_loss = calculate_interval_loss(min_PHI, max_PHI, xhat_PHI)

    return 0.1*bl_loss + 0.1*(curvature_loss + phi_loss)
