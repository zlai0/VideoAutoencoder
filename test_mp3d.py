import argparse
import os
import time
import logger
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from parser import test_mp3d_parser
import data.image_folder as D
import data.data_loader as DL
from models.autoencoder import *
from test_helper import *
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

parser = test_mp3d_parser()
args = parser.parse_args()
np.set_printoptions(precision=3)

def gettime():
    # get GMT time in string
    return time.strftime("%m%d%H%M", time.gmtime())

def main():
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    args.savepath = args.savepath+f'/test_mp3d_{gettime()}'
    log = logger.setup_logger(args.savepath + '/testing.log')

    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    TestData, _ = D.dataloader(args.dataset, 1, args.interval,
                               is_train=args.train_set, n_valid=0, load_all_frames=True)
    TestLoader = DataLoader(DL.ImageFloder(TestData, args.dataset),
                            batch_size=1, shuffle=False, num_workers=0)

    # get auto-encoder
    encoder_3d = Encoder3D(args)
    encoder_traj = EncoderTraj(args)
    rotate = Rotate(args)
    decoder = Decoder(args)

    # cuda
    encoder_3d = nn.DataParallel(encoder_3d).cuda()
    encoder_traj = nn.DataParallel(encoder_traj).cuda()
    rotate = nn.DataParallel(rotate).cuda()
    decoder = nn.DataParallel(decoder).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            encoder_3d.load_state_dict(checkpoint['encoder_3d'])
            encoder_traj.load_state_dict(checkpoint['encoder_traj'])
            decoder.load_state_dict(checkpoint['decoder'])
            rotate.load_state_dict(checkpoint['rotate'])
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')

    start_full_time = time.time()
    with torch.no_grad():
        log.info('start testing.')
        test(TestData, TestLoader, encoder_3d, encoder_traj, decoder, rotate, log)
    log.info('full testing time = {:.2f} Minutes'.format((time.time() - start_full_time) / 60))

def test(data, dataloader, encoder_3d, encoder_traj, decoder, rotate, log):
    _loss = AverageMeter()
    video_limit = min(args.video_limit, len(dataloader))
    frame_limit = args.frame_limit
    for b_i, video_clips in tqdm(enumerate(dataloader)):
        if b_i == video_limit: break

        encoder_3d.eval()
        encoder_traj.eval()
        decoder.eval()
        rotate.eval()

        clip = video_clips[0,:frame_limit].cuda()
        t, c, h, w = clip.size()

        preds = []
        scene_rep = encoder_3d(video_clips[:, 0])
        clip_in = torch.stack([clip[0], clip[1]])
        pose = get_pose_window(encoder_traj, clip_in)
        z = euler2mat(pose[1:])
        rot_codes = rotate(scene_rep, z)
        output = decoder(rot_codes)
        pred = F.interpolate(output, (h, w), mode='bilinear')
        pred = torch.clamp(pred, 0, 1)

        # output
        save_dir = os.path.join(args.savepath, f"Images/Seq_{str(b_i).zfill(4)}")
        os.makedirs(save_dir, exist_ok=True)
        torchvision.utils.save_image(pred, save_dir+'/output_image_.png')
        torchvision.utils.save_image(clip[0], save_dir+'/input_image_.png')
        torchvision.utils.save_image(clip[1], save_dir+'/tgt_image_.png')

    print()

if __name__ == '__main__':
    main()