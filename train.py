import os
import time
import logger
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from parser import train_parser
import data.image_folder as D
import data.data_loader as DL
from models.autoencoder import *
from models.discriminator import *
from train_helper import *
from eval_syn_re10k import compute_error_video
import numpy as np

parser = train_parser()
args = parser.parse_args()

def main():
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    log = logger.setup_logger(args.savepath + '/training.log')
    writer = SummaryWriter(log_dir=args.savepath)

    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    TrainData, _ = D.dataloader(args.dataset, args.clip_length, args.interval)
    _, ValidData = D.dataloader(args.dataset, args.clip_length, args.interval, load_all_frames=True)
    log.info(f'#Train vid: {len(TrainData)}')

    TrainLoader = DataLoader(DL.ImageFloder(TrainData, args.dataset),
        batch_size=args.bsize, shuffle=True, num_workers=args.worker,drop_last=True
    )
    ValidLoader = DataLoader(DL.ImageFloder(ValidData, args.dataset),
        batch_size=1, shuffle=False, num_workers=0,drop_last=True
    )

    # get auto-encoder
    encoder_3d = Encoder3D(args)
    encoder_traj = EncoderTraj(args)
    rotate = Rotate(args)
    decoder = Decoder(args)

    # get discriminator
    netd = NetD(args)

    # cuda
    encoder_3d = nn.DataParallel(encoder_3d).cuda()
    encoder_traj = nn.DataParallel(encoder_traj).cuda()
    rotate = nn.DataParallel(rotate).cuda()
    decoder = nn.DataParallel(decoder).cuda()
    netd = nn.DataParallel(netd).cuda()

    all_param = list(encoder_traj.parameters()) + list(encoder_3d.parameters()) + \
                list(decoder.parameters()) + list(rotate.parameters())

    optimizer_g = torch.optim.Adam(all_param, lr=args.lr, betas=(0,0.999))
    optimizer_d = torch.optim.Adam(netd.parameters(), lr=args.d_lr, betas=(0,0.999))

    log.info('Number of parameters: {}'.format(sum([p.data.nelement() for p in all_param])))

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            encoder_3d.load_state_dict(checkpoint['encoder_3d'],strict=False)
            encoder_traj.load_state_dict(checkpoint['encoder_traj'],strict=False)
            decoder.load_state_dict(checkpoint['decoder'],strict=False)
            rotate.load_state_dict(checkpoint['rotate'],strict=False)
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')

    start_full_time = time.time()

    for epoch in range(args.epochs):
        log.info('This is {}-th epoch'.format(epoch))
        train(TrainLoader, ValidLoader,
              encoder_3d, encoder_traj, rotate, decoder, netd,
              optimizer_g, optimizer_d, log, epoch, writer)

    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))

cur_max_psnr = 0
def train(TrainLoader, ValidLoader,
          encoder_3d, encoder_traj, rotate, decoder, netd,
          optimizer_g, optimizer_d, log, epoch, writer):
    _loss = AverageMeter()
    n_b = len(TrainLoader)
    torch.cuda.synchronize()
    b_s = time.perf_counter()

    for b_i, vid_clips in enumerate(TrainLoader):
        encoder_3d.train()
        encoder_traj.train()
        decoder.train()
        rotate.train()

        adjust_lr(args, optimizer_g, epoch, b_i, n_b)
        adjust_lr(args, optimizer_d, epoch, b_i, n_b)
        vid_clips = vid_clips.cuda()
        n_iter = b_i + n_b * epoch

        optimizer_g.zero_grad()

        l_r, fake_clips = compute_reconstruction_loss(args, encoder_3d, encoder_traj,
                                                       rotate, decoder, vid_clips, return_output=True)
        l_c = compute_consistency_loss(args, encoder_3d, encoder_traj, vid_clips)
        l_g = compute_gan_loss(netd, fake_clips)
        sum_loss = l_r + args.lambda_voxel * l_c + args.lambda_gan * l_g
        sum_loss.backward()
        optimizer_g.step()

        # train netd
        optimizer_d.zero_grad()
        l_d = train_netd(args, netd, vid_clips, fake_clips, optimizer_d)

        _loss.update(sum_loss.item())
        batch_time = time.perf_counter() - b_s
        b_s = time.perf_counter()

        writer.add_scalar('Reconstruction Loss (Train)', l_r, n_iter)
        info = 'Loss Image = {:.3f}({:.3f})'.format(_loss.val, _loss.avg) if _loss.count > 0 else '..'
        log.info('Epoch{} [{}/{}] {} T={:.2f}'.format(epoch, b_i, n_b, info, batch_time))

        if n_iter > 0 and n_iter % args.valid_freq == 0:
            with torch.no_grad():
                _ = test_reconstruction(ValidLoader, encoder_3d, encoder_traj, decoder, rotate,
                                    log, epoch, n_iter, writer)
                output_dir = visualize_synthesis(args, ValidLoader, encoder_3d, encoder_traj,
                                                 decoder, rotate, log, n_iter)
                avg_psnr, _, _ = test_synthesis(output_dir)

            log.info("Saving new checkpoint.")
            savefilename = args.savepath + '/checkpoint.tar'
            save_checkpoint(encoder_3d, encoder_traj, rotate, decoder, savefilename)
            global cur_max_psnr
            if avg_psnr > cur_max_psnr:
                log.info("Saving new best checkpoint.")
                cur_max_psnr = avg_psnr
                savefilename = args.savepath + '/checkpoint_best.tar'
                save_checkpoint(encoder_3d, encoder_traj, rotate, decoder, savefilename)

def test_reconstruction(dataloader, encoder_3d, encoder_traj, decoder, rotate, log, epoch, n_iter, writer):
    _loss = AverageMeter()
    n_b = len(dataloader)
    for b_i, vid_clips in enumerate(dataloader):
        encoder_3d.eval()
        encoder_traj.eval()
        decoder.eval()
        rotate.eval()
        b_s = time.perf_counter()
        vid_clips = vid_clips.cuda()
        with torch.no_grad():
            l_r = compute_reconstruction_loss(args, encoder_3d, encoder_traj,
                                              rotate, decoder, vid_clips)
        writer.add_scalar('Reconstruction Loss (Valid)', l_r, n_iter)
        _loss.update(l_r.item())
        info = 'Loss = {:.3f}({:.3f})'.format(_loss.val, _loss.avg)
        b_t = time.perf_counter() - b_s
        log.info('Validation at Epoch{} [{}/{}] {} T={:.2f}'.format(
            epoch, b_i, n_b, info, b_t))
    return _loss.avg

def test_synthesis(output_dir):
    values_psnr, values_ssim, values_lpips = [], [], []
    for i in range(20):
        video1 = output_dir+'/eval_video_{}_pred.mp4'.format(i)
        video2 = output_dir+'/eval_video_{}_true.mp4'.format(i)
        results = compute_error_video(video1, video2, lpips=False)
        values_psnr.append(results['PSNR'])
        values_ssim.append(results['SSIM'])
        values_lpips.append(results['LPIPS'])

    avg_psnr = np.mean(np.array(values_psnr))
    avg_ssim = np.mean(np.array(values_ssim))
    avg_lpips = np.mean(np.array(values_lpips))

    return (avg_psnr, avg_ssim, avg_lpips)

if __name__ == '__main__':
    main()