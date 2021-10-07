import torch
from models.util import euler2mat

def get_pose0(theta, im):
    # im: c, h, w
    imgs_pair = torch.cat([im[None], im[None]], dim=1) # t x 6 x h x w
    poses = theta(imgs_pair)
    P = euler2mat(poses)
    return P

def get_relative_pose(theta, im1, im2):
    imgs_src = im1[None].repeat(2,1,1,1)
    imgs_tgt = torch.stack([im1,im2])
    imgs_pair = torch.cat([imgs_src, imgs_tgt], dim=1) # t x 6 x h x w
    poses = theta(imgs_pair)
    Rt0 = euler2mat(poses[0:1])
    Rt1 = euler2mat(poses[1:2])
    R0 = Rt0[...,:3]
    R1 = Rt1[...,:3]
    t0 = Rt0[...,3]
    t1 = Rt1[...,3]
    R = R0.transpose(1,2) @ R1
    t = R0.transpose(1,2) @ (t1-t0).unsqueeze(2)
    P = torch.cat([R,t], dim=2)
    return P

def get_poses(theta, img_in):
    t, c, h, w = img_in.size()
    poses = [get_pose0(theta,img_in[0])]
    for i in range(t-1):
        p = get_relative_pose(theta,img_in[i],img_in[i+1])
        poses.append(p)
    poses = torch.cat(poses)
    return poses

def get_pose_window(theta, img_in):
    t, c, h, w = img_in.size()
    imgs_ref = img_in[0:1].repeat(t,1,1,1)
    imgs_pair = torch.cat([imgs_ref, img_in], dim=1) # t x 6 x h x w
    pair_tensor = imgs_pair.view(t, c*2, h, w)
    poses = theta(pair_tensor)
    return poses

def construct_trajectory(poses):
    # poses: t, 3, 4
    # return: t, 3, 4
    cur_t = torch.tensor([0,0,0]).to(poses.device)
    cur_R = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]).to(poses.device)

    cur_ts, cur_Rs = [], []
    for pose in poses:
        r = pose[:,:3]
        t = pose[:,3]
        cur_t = cur_t + cur_R @ t
        cur_R = r @ cur_R
        cur_ts.append(cur_t)
        cur_Rs.append(cur_R)

    R = torch.stack(cur_Rs)
    t = torch.stack(cur_ts)
    P = torch.cat([R,t.unsqueeze(2)], dim=2)
    return P

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
