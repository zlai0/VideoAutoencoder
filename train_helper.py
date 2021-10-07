import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision.io as io
from models.util import euler2mat
from models.submodule import VGGPerceptualLoss, stn

perceptual_loss = VGGPerceptualLoss()
perceptual_loss = nn.DataParallel(perceptual_loss).cuda()

def compute_reconstruction_loss(args, encoder_3d, encoder_traj, rotate, decoder, clips, return_output=False):
    b, t, c, h, w = clips.size()
    codes = encoder_3d(clips[:,0])
    _,C,H,W,D = codes.size()
    code_t = codes.unsqueeze(1).repeat(1, t, 1, 1, 1, 1).view(b * t, C, H, W, D)

    clips_ref = clips[:,0:1].repeat(1,t,1,1,1)
    clips_pair = torch.cat([clips_ref, clips], dim=2)
    pair_tensor = clips_pair.view(b*t, c*2, h, w)
    poses = encoder_traj(pair_tensor)
    theta = euler2mat(poses)

    rot_codes = rotate(code_t, theta)
    output = decoder(rot_codes)

    output = F.interpolate(output, (h, w), mode='bilinear')
    target = clips.view(b*t, c, h, w)
    loss_perceptual = perceptual_loss(output, target)
    loss_l1 = (output - target).abs().mean()
    loss = loss_perceptual.mean() * args.lambda_perc + loss_l1 * args.lambda_l1
    if return_output:
        return loss, output
    else:
        return loss

def compute_consistency_loss(args, encoder_3d, encoder_traj, clips):
    b, t, c, h, w = clips.size()
    code0 = encoder_3d(clips[:,0])
    _,C,H,W,D = code0.size()

    clips_pair = torch.cat([clips[:,:-1], clips[:,1:]], dim=2)
    pair_tensor = clips_pair.view(b*(t-1), c*2, h, w)
    poses = encoder_traj(pair_tensor)  # b*t-1 x 6
    theta = euler2mat(poses).reshape(b, t-1, 3, 4)  # b x 3 x 4

    code = code0
    codes = []
    for i in range(t-1):
        code = stn(code, theta[:, i])
        codes.append(code)

    codes = torch.stack(codes, dim=1).reshape(b*(t-1), C, H, W, D)
    target = encoder_3d(clips[:,1:].reshape(b*(t-1), c, h, w))

    loss = (codes-target).abs().mean()
    return loss

def compute_gan_loss(netd, fake):
    netd.zero_grad()
    output = netd(fake)
    error_G = -output.mean()
    return error_G

def train_netd(args, netd, real_images, fake_images, optimizer_d):
    b, t, c, h, w = real_images.size()
    netd.zero_grad()
    ## train netd with real poses
    img_tensor = real_images.view(b*t, c, h, w)
    real = img_tensor.detach()  # only use n_gan_angles images
    output = netd(real)
    real_predict1 = output.mean() - 0.001 * (output ** 2).mean()
    error_real = -real_predict1
    error_real.backward()
    ## train netd with fake poses
    img_tensor = fake_images.view(b*t, c, h, w)
    fake = img_tensor.detach()  # only use n_gan_angles images
    output2 = netd(fake)
    error_fake = output2.mean()
    error_fake.backward()
    # calculate gradient penalty
    eps = torch.rand(b*t, 1, 1, 1).cuda()
    x_hat = eps * real.data + (1 - eps) * fake.data
    x_hat.requires_grad = True
    hat_predict = netd(x_hat)
    grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
    grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
    grad_penalty = 10 * grad_penalty
    grad_penalty.backward()
    ################################
    error_D = error_real + error_fake
    optimizer_d.step()
    return error_D


def get_pose_window(theta, clip_in):
    t, c, h, w = clip_in.size()
    clips_ref = clip_in[0:1].repeat(t,1,1,1)
    clips_pair = torch.cat([clips_ref, clip_in], dim=1) # t x 6 x h x w
    pair_tensor = clips_pair.view(t, c*2, h, w)
    poses = theta(pair_tensor)
    return poses

def visualize_synthesis(args, dataloader, encoder_3d, encoder_traj, decoder, rotate, log, n_iter):
    n_b = len(dataloader)
    n_eval_video = 20
    scene_update_freq = 12
    for b_i, vid_clips in enumerate(dataloader):
        encoder_3d.eval(); decoder.eval(); rotate.eval(); encoder_traj.eval();
        fend = 30
        vid_clips = vid_clips.cuda()[:,:fend]
        b, t, c, h, w = vid_clips.size()
        clips = vid_clips.view(b * t, c, h, w)
        preds = []
        for i in range(t-1):
            if i == 0:
                preds = []
                preds.append(clips[0:1])
                scene_rep = encoder_3d(vid_clips[:, 0])  # Only use T=0. Size: B x c x h x w x d
                scene_index = 0
            elif i % scene_update_freq == 0:
                scene_rep = encoder_3d(pred)  # Update scene representation
                scene_index = i
            clips_in = torch.stack([clips[scene_index], clips[i+1]])
            pose = get_pose_window(encoder_traj, clips_in)   # 2, 6
            z = euler2mat(pose[1:])
            rot_codes = rotate(scene_rep, z)
            output = decoder(rot_codes)
            pred = F.interpolate(output, (h, w), mode='bilinear')  # T*B x 3 x H x W
            pred = torch.clamp(pred, 0, 1)
            preds.append(pred)
        preds = torch.cat(preds,dim=0)
        if b_i == n_eval_video:
            log.info('Inference finished.')
            break
        ###### Output
        save_dir = os.path.join(args.savepath, 'eval_videos', f"iter_{n_iter//1000}k")
        os.makedirs(save_dir, exist_ok=True)
        vid = (clips.permute(0,2,3,1) * 255).byte().cpu()
        io.write_video(save_dir+'/eval_video_{}_true.mp4'.format(b_i), vid, 6)
        pred = (preds.permute(0,2,3,1) * 255).byte().cpu()
        io.write_video(save_dir+'/eval_video_{}_pred.mp4'.format(b_i), pred, 6)

    return save_dir

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

def adjust_lr(args, optimizer, epoch, batch, n_b):
    iteration = batch + epoch * n_b

    if iteration <= 80000 * args.lr_adj:
        lr = args.lr
    elif iteration <= 120000 * args.lr_adj:
        lr = args.lr * 0.5
    elif iteration <= 160000 * args.lr_adj:
        lr = args.lr * 0.25
    elif iteration <= 200000 * args.lr_adj:
        lr = args.lr * 0.125
    else:
        lr = args.lr * 0.0625

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(encoder_3d, encoder_traj, rotate, decoder, savefilename):
    torch.save({
        'encoder_3d': encoder_3d.state_dict(),
        'encoder_traj': encoder_traj.state_dict(),
        'rotate': rotate.state_dict(),
        'decoder': decoder.state_dict(),
    }, savefilename)

