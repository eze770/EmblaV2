import numpy as np
from envs import GymPixelsProcessingWrapper, CleanGymWrapper
import gymnasium as gym
from gymnasium.wrappers import AddRenderObservation
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
import cv2
from typing import Optional, Tuple, List
from torch.amp import autocast
import torch.nn.functional as F
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def crop_center(
        img: torch.Tensor,
        frac: float = 0.5
) -> torch.Tensor:
    r"""
  Crop center square from image.
  """
    h_offset = round(img.shape[1] * (frac / 2))
    w_offset = round(img.shape[2] * (frac / 2))
    return downscale(img[..., h_offset:-h_offset, w_offset:-w_offset], 16)


def downscale(img, size=16):
    """
    img: [B, C, H, W] or [C, H, W]
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)

    img = F.interpolate(
        img,
        size=(size, size),
        mode="bilinear",
        align_corners=False
    )
    return img


def color_filter(config, img):
    img = (img * 255).astype(np.uint8)
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # lower = np.array([280, 0.3, 0.3], dtype=np.float32)
    # upper = np.array([340, 255, 255], dtype=np.float32)
    lower = np.array([140, 80, 80], dtype=np.uint8)  # 0, 0, 0 for black (eze)
    upper = np.array([170, 255, 255], dtype=np.uint8)  # 180, 255, 60 for black (eze)
    mask = cv2.inRange(hsvImg, lower, upper)
    kernel = np.ones((config.selfModel.filterSize, config.selfModel.filterSize),
                     np.uint8)  # filter 1 by 1 pixels, (eze)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    else:
        clean_mask = np.zeros_like(mask)
    maskedImg = (torch.from_numpy(clean_mask).to(device).float() > 0.5) * 10  # *10 because 0 (not robot) would be to dominant otherwise, (eze)
    return maskedImg


def rot_X(th: float) -> np.ndarray:
    """Creates a 4x4 rotation matrix around the X-axis."""
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(th), -np.sin(th), 0],
        [0, np.sin(th), np.cos(th), 0],
        [0, 0, 0, 1]
    ])

def rot_Y(th: float) -> np.ndarray:
    """Creates a 4x4 rotation matrix around the Y-axis."""
    return np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]
    ])

def rot_Z(th: float) -> np.ndarray:
    """Creates a 4x4 rotation matrix around the Z-axis."""
    return np.array([
        [np.cos(th), -np.sin(th), 0, 0],
        [np.sin(th), np.cos(th), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def pts_trans_matrix_numpy(theta,phi,no_inverse=False):
    # the coordinates in pybullet, camera is along X axis, but in the pts coordinates, the camera is along z axis

    w2c = transition_matrix("rot_z", -theta / 180. * np.pi)
    w2c = np.dot(transition_matrix("rot_y", -phi / 180. * np.pi), w2c)
    if no_inverse == False:
        w2c = np.linalg.inv(w2c)
    return w2c


def pts_trans_matrix(theta, phi, no_inverse=False):
    # the coordinates in pybullet, camera is along X axis,
    # but in the pts coordinates, the camera is along z axis

    w2c = transition_matrix_torch("rot_z", -theta / 180. * torch.pi)
    w2c = transition_matrix_torch("rot_y", -phi / 180. * torch.pi) @ w2c
    if not no_inverse:
        dtype = w2c.dtype
        w2c = torch.inverse(w2c.float()).to(dtype)
    return w2c


def rays_np(H, W, D, c_h=1.106):
    """numpy version my_ray"""
    rate = np.tan(21 * np.pi / 180)
    # co = torch.Tensor([0.8, 0, 0.606])
    #
    #               0.3          0.3        0.5
    #         far -----  object ----- near ----- camera
    #
    near = np.array([
        [[0.3, 0.5 * rate, c_h + 0.5 * rate], [0.3, -0.5 * rate, c_h + 0.5 * rate]],
        [[0.3, 0.5 * rate, c_h - 0.5 * rate], [0.3, -0.5 * rate, c_h - 0.5 * rate]]
    ])

    far = np.array([
        [[-0.3, 1.1 * rate, c_h + 1.1 * rate], [-0.3, -1.1 * rate, c_h + 1.1 * rate]],
        [[-0.3, 1.1 * rate, c_h - 1.1 * rate], [-0.3, -1.1 * rate, c_h - 1.1 * rate]]
    ])
    n_y_list = (np.linspace(near[0, 0, 1], near[0, 1, 1], W + 1) + 0.5 * (near[0, 1, 1] - near[0, 0, 1]) / W)[:-1]
    n_z_list = (np.linspace(near[0, 0, 2], near[1, 0, 2], H + 1) + 0.5 * (near[1, 0, 2] - near[0, 0, 2]) / H)[:-1]
    f_y_list = (np.linspace(far[0, 0, 1], far[0, 1, 1], W + 1) + 0.5 * (far[0, 1, 1] - far[0, 0, 1]) / W)[:-1]
    f_z_list = (np.linspace(far[0, 0, 2], far[1, 0, 2], H + 1) + 0.5 * (far[1, 0, 2] - far[0, 0, 2]) / H)[:-1]

    ny, nz = np.meshgrid(n_y_list, n_z_list)
    near_face = np.stack([0.3 * np.ones_like(ny.T), ny.T, nz.T], -1)

    fy, fz = np.meshgrid(f_y_list, f_z_list)
    far_face = np.stack([-0.3 * np.ones_like(fy.T), fy.T, fz.T], -1)
    D_list = np.linspace(0, 1, D + 1)[:-1] + .5 * (1 / D)
    box = []
    for d in D_list:
        one_face = (near_face - far_face) * d + far_face
        box.append(one_face)

    box = np.array(box)
    box = np.swapaxes(box, 0, 2)
    # box = torch.swapaxes(box, 1, 2)
    return near, far, near_face, far_face, box


def transfer_box(vbox, norm_angles, c_h=1.106, forward_flag=False):
    vb_shape = vbox.shape
    flatten_box = vbox.reshape(vb_shape[0] * vb_shape[1] * vb_shape[2], 3)
    flatten_box[:, 2] -= c_h
    full_matrix = np.dot(rot_Z(norm_angles[0] * 360 / 180 * np.pi), rot_Y(norm_angles[1] * 90 / 180 * np.pi))
    if forward_flag:
        # static arm, moving camera
        flatten_new_view_box = np.dot(
            full_matrix,
            np.hstack((flatten_box, np.ones((flatten_box.shape[0], 1)))).T
        )[:3]
    else:
        # static camera, moving arm
        flatten_new_view_box = np.dot(
            np.linalg.inv(full_matrix),
            np.hstack((flatten_box, np.ones((flatten_box.shape[0], 1)))).T
        )[:3]
    flatten_new_view_box[2] += c_h
    flatten_new_view_box = flatten_new_view_box.T
    new_view_box = flatten_new_view_box.reshape(vb_shape[0], vb_shape[1], vb_shape[2], 3)
    return new_view_box, flatten_new_view_box


def get_rays(
        height: int,
        width: int,
        focal_length: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Find origin and direction of rays through every pixel and camera origin.

    # Apply pinhole camera model to gather directions at each pixel
    # i, j = torch.meshgrid(
    #     torch.arange(width, dtype=torch.float32).to(focal_length),
    #     torch.arange(height, dtype=torch.float32).to(focal_length),
    #     indexing='ij')

    # debug jiong @ Aug 26, to(focal_length) is to focal_length's device, transfered again after get_rays function
    
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device),
        indexing='ij')

    directions = torch.stack([(i - width * .5) / focal_length,
                              -(j - height * .5) / focal_length,
                              -torch.ones_like(i)
                              ], dim=-1)
    # directions: tan_i, tan_j, -1

    # Apply camera pose to directions
    rays_d = directions
    rays_o = torch.tensor([1, 0, 0], dtype=torch.float32, device=device)
    rays_o = rays_o.expand(directions.shape)

    rays_d_clone = rays_d.clone()
    rays_d[..., 0], rays_d[..., 2] = rays_d_clone[..., 2].clone(), rays_d_clone[..., 0].clone()

    # Origin is same for all directions (the optical center)
    rotation_matrix = torch.tensor([[1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, -1]], dtype=torch.float32, device=device)
    rotation_matrix = rotation_matrix[None, None].to(rays_d)

    # Rotate the points
    rays_d = torch.matmul(rays_d, rotation_matrix)
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    return rays_o, rays_d

def sample_stratified(
        rays_o: torch.Tensor,  # [N_rays, 3]
        rays_d: torch.Tensor,  # [N_rays, 3]
        arm_angle: torch.Tensor,  # [B, :]
        near: float,
        far: float,
        n_samples: int,
        perturb: Optional[bool] = True,
        inverse_depth: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    B = arm_angle.shape[0]
    N_rays = rays_o.shape[0]

    # Grab samples for space integration along ray
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
    if not inverse_depth:
        # Sample linearly between `near` and `far`
        x_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity)
        x_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    # Draw uniform samples from bins along ray
    if perturb:
        mids = .5 * (x_vals[1:] + x_vals[:-1])
        upper = torch.concat([mids, x_vals[-1:]], dim=0)
        lower = torch.concat([x_vals[:1], mids], dim=0)
        t_rand = torch.rand([n_samples], device=device)
        x_vals = lower + (upper - lower) * t_rand

    # [1, N_rays, n_samples]
    x_vals = x_vals.view(1, 1, n_samples).expand(B, N_rays, n_samples)

    # [1, N_rays, 3]
    rays_o = rays_o.unsqueeze(0)
    rays_d = rays_d.unsqueeze(0)

    # [B, N_rays, n_samples, 3]
    pts = rays_o[..., None, :] + rays_d[..., None, :] * x_vals[..., :, None]

    # Transformationsmatrix
    pose_matrix = pts_trans_matrix(arm_angle[:, 0], arm_angle[:, 1]).to(device)

    # [B, 3, 3]
    R = pose_matrix[:, :3, :3]

    # Batch Matrix Multiply
    pts = torch.matmul(pts, R[:, None, :, :])

    return pts, x_vals



"""
volume rendering
"""
def VR_rendering(
        raw: torch.Tensor,
        x_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    dense = 1.0 - torch.exp(-nn.functional.relu(raw[..., 0]))

    render_img = torch.sum(dense, dim=-1)

    return render_img, dense

def VRAT_rendering(
        raw: torch.Tensor,
        x_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    dists = x_vals[..., 1:] - x_vals[..., :-1]

    # add one elements for each ray to compensate the size to 64
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1).to(device)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    alpha_dense = 1.0 - torch.exp(-nn.functional.relu(raw[..., 0]) * dists)

    render_img = torch.sum(alpha_dense, dim=-1)

    return render_img, alpha_dense

def OM_rendering(
        raw: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 1]))
    rgb_each_point = alpha*raw[..., 0]
    render_img = torch.sum(rgb_each_point, dim=-1)

    return render_img, alpha

def OM_rendering_split_output(raw):
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 1]))
    rgb_each_point = alpha*raw[..., 0]
    render_img = torch.sum(rgb_each_point, dim=-1)
    visibility = raw[..., 0]
    return render_img, alpha, visibility

def sample_pdf(
        bins: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        perturb: bool = False
) -> torch.Tensor:
    r"""
  Apply inverse transform sampling to a weighted set of points.
  """

    # Normalize weights to get PDF.
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True)  # [n_rays, weights.shape[-1]]

    # Convert PDF to CDF.
    cdf = torch.cumsum(pdf, dim=-1)  # [n_rays, weights.shape[-1]]
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [n_rays, weights.shape[-1] + 1]

    # Take sample positions to grab from CDF. Linear when perturb == 0.
    if not perturb:
        u = torch.linspace(0., 1., n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])  # [n_rays, n_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device)  # [n_rays, n_samples]

    # Find indices along CDF where values in u would be placed.
    u = u.contiguous()  # Returns contiguous tensor with same values.
    inds = torch.searchsorted(cdf, u, right=True)  # [n_rays, n_samples]

    # Clamp indices that are out of bounds.
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)  # [n_rays, n_samples, 2]

    # Sample from cdf and the corresponding bin centers.
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                         index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                          index=inds_g)

    # Convert samples to ray length.
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples  # [n_rays, n_samples]


def sample_hierarchical(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        perturb: bool = False,
        angle: float = 1.,
        more_dof: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
  Apply hierarchical sampling to the rays.
  """

    # Draw samples from PDF using z_vals as bins and weights as probabilities.
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples,
                               perturb=perturb)
    new_z_samples = new_z_samples.detach()

    # Resample points from ray based on PDF.
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :,
                                                        None]  # [N_rays, N_samples + n_samples, 3]
    if more_dof:
        # for 3dof arm
        add_angle = torch.ones(pts.shape[0], pts.shape[1], 1).to(device) * angle
        pts = torch.cat((pts, add_angle), 2)
        # print(pts.shape)
    return pts, z_vals_combined, new_z_samples


def prepare_chunks(
        points: torch.Tensor,
        chunksize: int = 2 ** 14
) -> List[torch.Tensor]:

    points = points.reshape((-1, points.shape[-1]))
    points = [points[i:i + chunksize] for i in range(0, points.shape[0], chunksize)]
    return points


def self_model_forward(
        config,
        model: nn.Module,
        arm_angle: torch.Tensor,
        output_flag: int = 0,
        observation_shape=None
):
    config = config.dreamer
    Camera_FOV = config.selfModel.cameraFOV
    height, width = observation_shape[1]/4, observation_shape[2]/4
    camera_angle_y = Camera_FOV * np.pi / 180.
    focal = 0.5 * height / np.tan(0.5 * camera_angle_y)
    focal = torch.tensor(focal, dtype=torch.float32, device="cuda")

    rays_o, rays_d = get_rays(int(height), int(width), focal)
    DOF = config.selfModel.dof
    cam_dist = config.selfModel.camDist
    nf_size = config.selfModel.nfSize
    near, far = cam_dist - nf_size, cam_dist + nf_size  # real scale dist=1.0
    n_samples = config.selfModel.nSamples
    chunksize = eval(config.selfModel.chunkSize)  # Modify as needed to fit in GPU memory

    return_outputs = True
    if output_flag == 4:  # 4 is mode 0 with latent output, (eze)
        output_flag = 0
        return_outputs = False

    # Sample query points along each ray.
    # query_points: [B, N_rays, N_samples, 3]
    # z_vals:       [B, N_rays, N_samples]
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, arm_angle, near, far, n_samples=n_samples)
    # Prepare batches.
    B, N_rays, N_samples, _ = query_points.shape

    # arm_angle = arm_angle / 180 * np.pi  # not used here because angles already are in rad, (eze)
    if DOF > 2:
        extra = arm_angle[:, 2:DOF]  # [B, DOF-2]

        extra = extra[:, None, None, :]  # [B,1,1,DOF-2]
        extra = extra.expand(B, N_rays, N_samples, DOF-2)
        model_input = torch.cat((query_points, extra), dim=-1)

    # arm_angle[:DOF] -> use one angle
    else:
        model_input = query_points  # orig version 3 input 2dof, Mar30

    batches = prepare_chunks(model_input, chunksize=chunksize)

    predictions = torch.zeros(len(batches), batches[0].shape[0], 2, device=device)
    latent_states = torch.zeros(len(batches), batches[0].shape[0], config.selfModel.d_filter//4, device=device)
    with autocast("cuda"):
        c = 0
        for batch in batches:
            if return_outputs:
                prediction, latent_state = model(batch)
                predictions[c] = prediction
            else:
                latent_state = model(batch)
            latent_states[c] = latent_state
            del batch
            c += 1

    raw = predictions.reshape(B, N_rays, N_samples, -1)

    # rays_d zu Batch broadcasten
    rays_d_batch = rays_d.unsqueeze(0).expand(B, -1, -1)

    if output_flag ==0:
        rgb_map, rgb_each_point = OM_rendering(raw)
    elif output_flag ==1:
        rgb_map, rgb_each_point = VR_rendering(raw, z_vals, rays_d_batch)
    elif output_flag ==2:
        rgb_map, rgb_each_point = VRAT_rendering(raw, z_vals, rays_d_batch)
    elif output_flag ==3:
        rgb_map,rgb_each_point, visibility = OM_rendering_split_output(raw)
        return rgb_map, query_points, rgb_each_point, visibility

    outputs = {
        'rgb_map': rgb_map,
        'rgb_each_point': rgb_each_point,
        'query_points': query_points}

    # Store outputs.
    latent_state = latent_states.view(B, -1, latent_states.shape[-1]).mean(dim=1)
    if return_outputs:
        return latent_state, outputs
    else:
        return latent_state


# ---------------------------------------------------------
# Transformation Matrices for 3D Space
# ---------------------------------------------------------
def transition_matrix(label: str, value: float) -> np.ndarray:
    """Returns a 4x4 transformation matrix for rotation in 3D space."""
    if label == "rot_x":
        return rot_X(value)
    elif label == "rot_y":
        return rot_Y(value)
    elif label == "rot_z":
        return rot_Z(value)
    else:
        raise ValueError("Invalid label. Use 'rot_x', 'rot_y', or 'rot_z'.")


def transition_matrix_torch(label: str, value: torch.Tensor) -> torch.Tensor:
    """Returns a 4x4 transformation matrix for rotation in 3D space using PyTorch tensors."""
    matrix = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(value.shape[0],1,1)

    if label == "rot_x":
        matrix[:, 1, 1] = torch.cos(value)
        matrix[:, 1, 2] = -torch.sin(value)
        matrix[:, 2, 1] = torch.sin(value)
        matrix[:, 2, 2] = torch.cos(value)
    elif label == "rot_y":
        matrix[:, 0, 0] = torch.cos(value)
        matrix[:, 0, 2] = -torch.sin(value)
        matrix[:, 2, 0] = torch.sin(value)
        matrix[:, 2, 2] = torch.cos(value)
    elif label == "rot_z":
        matrix[:, 0, 0] = torch.cos(value)
        matrix[:, 0, 1] = -torch.sin(value)
        matrix[:, 1, 0] = torch.sin(value)
        matrix[:, 1, 1] = torch.cos(value)
    else:
        raise ValueError("Invalid label. Use 'rot_x', 'rot_y', or 'rot_z'.")

    return matrix

def plot_3d_visual(x, y, z, if_transform=True):
    if if_transform:
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        z = z.detach().cpu().numpy()

    ax = plt.axes(projection='3d')
    ax.scatter3D(x,
                 y,
                 z, s=1
                 )
    # ax.scatter3D(0,0,0)


# ---------------------------------------------------------
# (eze)
# ---------------------------------------------------------
def init_envs(config):
    camMode = config.camMode
    smEnv             = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(AddRenderObservation(gym.make(config.environmentName, render_mode="rgb_array", max_episode_steps=None, camera_name="third_person", forward_reward_weight=0), render_only=True), (64, 64))))
    if camMode == 1:
        camName = "ego_cam"
        wmEnv         = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(AddRenderObservation(gym.make(config.environmentName, render_mode="rgb_array", max_episode_steps=None, camera_name=camName, forward_reward_weight=0), render_only=True), (64, 64))))
    elif camMode == 2:
        camName = "topdown_cam"
        wmEnv         = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(AddRenderObservation(gym.make(config.environmentName, render_mode="rgb_array", max_episode_steps=None, camera_name=camName, forward_reward_weight=0), render_only=True), (64, 64))))
    else:
        wmEnv = None

    return smEnv, wmEnv

# ---------------------------------------------------------
# From original sm Code (no usage here): (eze)
# ---------------------------------------------------------

if __name__ == "__main__":

    #              -0.4    0      0.4        0.6
    #         far -----  object ----- near ----- camera
    #

    DOF = 2  # the number of motors  # dof4 apr03
    num_data = 20**DOF
    pxs = 100  # collected data pixels

    HEIGHT = pxs
    WIDTH = pxs
    nf_size = 0.4
    cam_dist = 1
    camera_angle_x = 42 * np.pi / 180.
    focal = .5 * WIDTH / np.tan(.5 * camera_angle_x)
    rays_o, rays_d = get_rays(HEIGHT, WIDTH, focal)

    # Visualization
    ax = plt.figure().add_subplot(projection='3d')
    rays_o = rays_o.detach().cpu().numpy()
    rays_d = rays_d.detach().cpu().numpy()
    rays_d = rays_d[:10]
    idx = np.random.choice(list(np.arange(len(rays_d))),size = 1000)
    rays_d = rays_d[idx]
    rays_o = rays_o[idx]
    for plt_i in range(len(rays_d)):
        ax.plot3D([rays_d[plt_i, 0],0],
                  [rays_d[plt_i, 1],0],
                  [rays_d[plt_i, 2],0])
    print(rays_o)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(rays_o[0][0],rays_o[0][1],rays_o[0][2])
    plt.show()
    quit()

    data = np.load('data/sim_data/sim_data_robo0(arm).npz' )

    training_angles = torch.from_numpy(data['angles'].astype('float32'))
    training_pose_matrix = torch.from_numpy(data['poses'].astype('float32'))

    idxx = 265
    angle = training_angles[idxx]
    print(angle/90)
    pose_matrix = pts_trans_matrix(angle[0],angle[1])

    near, far = cam_dist - nf_size, cam_dist + nf_size
    kwargs_sample_stratified = {
        'n_samples': 64,
        'perturb': True,
        'inverse_depth': False
    }

    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3])
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, angle, near, far, **kwargs_sample_stratified)

