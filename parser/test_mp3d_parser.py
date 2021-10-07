import argparse

def test_parser():
    parser = argparse.ArgumentParser(description='Video Auto-encoder MP3D/Replica Testing Options')

    # experiment specifics
    parser.add_argument('--dataset', default='Matterport3D',
                        help='Name of the dataset')
    parser.add_argument('--savepath', type=str, default='log/test',
                        help='Path for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint file to resume')
    parser.add_argument('--worker', type=int, default=8,
                        help='number of dataloader threads')

    # model options
    parser.add_argument('--encoder_3d', type=str, default='conv',
                        help='3d encoder architecture')
    parser.add_argument('--encoder_traj', type=str, default='conv',
                        help='trajectory encoder architecture')
    parser.add_argument('--decoder', type=str, default='conv',
                        help='decoder style')
    parser.add_argument('--padding_mode', type=str, default='zeros',
                        help='grid sampling padding mode')
    parser.add_argument('--scale_rotate', type=float, default=0.01,
                        help='scale for rotation values')
    parser.add_argument('--scale_translate', type=float, default=0.01,
                        help='scale for translation values')

    # testing options
    parser.add_argument('--frame_limit', type=int, default=2,
                        help='number of videos frames to test')
    parser.add_argument('--video_limit', type=int, default=3000,
                        help='number of videos to test')
    parser.add_argument('--train_set', action='store_true',
                        help='use train set')
    parser.add_argument('--interval', type=int, default=1,
                        help='interval between clip frames')

    return parser
