# Adapted from https://github.com/facebookresearch/synsin/blob/master/evaluation/evaluate_perceptualsim.py
import argparse
from glob import glob
from tqdm import tqdm
from pprint import pprint
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.io as io
from util.metrics import perceptual_sim, psnr, ssim_metric
from util.pretrained_networks import PNet
transform = transforms.Compose([transforms.ToTensor()])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_videos(videoname):
    vid, _, _ = io.read_video(videoname, pts_unit='sec')
    vid = vid.permute(0,3,1,2) / 255.0
    return vid


def compute_perceptual_similarity(video1, video2, lpips=False):
    # Load VGG16 for feature similarity
    vgg16 = PNet(use_gpu=(device=='cuda'))
    vgg16.eval()

    assert video1.shape == video2.shape

    values_percsim = []
    values_ssim = []
    values_psnr = []
    video_length = len(video1)
    for i in range(video_length):
        perc_sim = 10000
        ssim_sim = -10
        psnr_sim = -10

        t_img = video1[i][None].contiguous().to(device)
        p_img = video2[i][None].contiguous().to(device)

        if lpips:
            t_perc_sim = perceptual_sim(p_img, t_img, vgg16).item()
        else:
            t_perc_sim = 0
        perc_sim = min(perc_sim, t_perc_sim)

        ssim_sim = max(ssim_sim, ssim_metric(p_img, t_img).item())
        psnr_sim = max(psnr_sim, psnr(p_img, t_img).item())

        values_percsim += [perc_sim]
        values_ssim += [ssim_sim]
        values_psnr += [psnr_sim]

    avg_percsim = np.mean(np.array(values_percsim))
    avg_psnr = np.mean(np.array(values_psnr))
    avg_ssim = np.mean(np.array(values_ssim))

    return {
        "LPIPS": avg_percsim,
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
    }

def compute_error_video(videopath1, videopath2, lpips):
    video1 = load_videos(videopath1)
    video2 = load_videos(videopath2)
    results = compute_perceptual_similarity(video1, video2, lpips=lpips)
    return results

def map_fn(i):
    est_file = f'{output_dir}/video_{i}_pred.mp4'
    ref_file = f'{output_dir}/video_{i}_vid.mp4'

    video1 = load_videos(est_file)
    video2 = load_videos(ref_file)

    results = compute_perceptual_similarity(video1, video2, lpips=args.lpips)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str, default=None)
    parser.add_argument('--lpips', action='store_true')
    args = parser.parse_args()
    output_dir = args.output_dir

    values_psnr, values_ssim, values_lpips = [], [], []

    max_i = len(glob(f'{output_dir}/video_*_true.mp4'))
    if max_i == 0:
        print('no videos found.')
        exit()
    print(f'Evaluating {max_i} trajectories.')

    pb = tqdm(range(max_i))
    for i in pb:
        est_file = f'{output_dir}/video_{i}_pred.mp4'
        ref_file = f'{output_dir}/video_{i}_true.mp4'

        video1 = load_videos(est_file)
        video2 = load_videos(ref_file)

        results = compute_perceptual_similarity(video1, video2, lpips=args.lpips)
        if not (np.isfinite(results['PSNR']) and np.isfinite(results['SSIM']) and np.isfinite(results['LPIPS'])):
            continue

        values_psnr.append(results['PSNR'])
        values_ssim.append(results['SSIM'])
        values_lpips.append(results['LPIPS'])

        avg_psnr = np.mean(np.array(values_psnr))
        avg_ssim = np.mean(np.array(values_ssim))
    avg_lpips = np.mean(np.array(values_lpips))
    pprint({'avg_psnr':avg_psnr, 'avg_ssim':avg_ssim, 'avg_lpips':avg_lpips})