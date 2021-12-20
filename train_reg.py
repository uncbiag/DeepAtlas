"""
Train a 3d voxel_morph_cvpr
"""
import os
import argparse

from models.registration import RegistrationExperiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-g', default='0', type=str,
                        help='index of used GPU')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='if debug mode')
    parser.add_argument('--preload', '-load', action='store_true',
                        help='if preload data into memory to speed up IO')
    parser.add_argument('--num_samples', '-ns', default=21, type=int,
                        help='number of samples for training')
    parser.add_argument('--num_epochs', '-ne', default=100, type=int,
                        help='number of samples for training')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--simE', default=1, type=int,
                        help='weight of image similarity loss')
    parser.add_argument('--regE', default=0.1, type=int,
                        help='weight of regularization loss')
    parser.add_argument('--segE', default=3, type=int,
                        help='weight of segmentation overlap loss')
    parser.add_argument('--test_only', '-t', action='store_true',
                        help='only test model')
    parser.add_argument('--data_root', '-root', default='/playpen-raid/zhenlinx/Data/brains/Mindboggle101/histogram_matched/', type=str,
                        help='root of the data folder')
    parser.add_argument('--data', default='MindBoggle', type=str,
                        help='which dataset to use')
    parser.add_argument('--fold', default=0, type=int,
                        help='training data fold')
    parser.add_argument('--log_root', '-log', default='./logs', type=str,
                        help='root of the log folders that saves logs/checkpoints')
    args = parser.parse_args()

    n_classes = 32

    config = dict(
        data=args.data,
        data_dir=os.path.join(args.data_root, "mindboggle"),
        valid_data_dir=os.path.join(args.data_root, "mindboggle"),

        num_training_samples=130,

        n_epochs=args.num_epochs,
        batch_size=1,
        valid_batch_size=1,
        valid_samples=90,
        train_data_loader_n_workers=4,
        valid_data_loader_n_workers=2,

        print_batch_period=100,

        valid_epoch_period=1,
        valid_batch_period=1000,
        save_ckpts_epoch_period=1,

        reg_model='voxel_morph_cvpr',  # UNet/UNet_light/UNet_light2
        reg_model_setting=dict(input_channel=2, output_channel=3,
                               enc_filters=[16, 32, 32, 32, 32],
                               dec_filters=[32, 32, 32, 8, 8]
                               ),
        n_classes=n_classes,
        class_name=[str(k) for k in range(1, n_classes)],
        n_channels=1,
        if_bias=1,
        if_batchnorm=0,

        crop_size=[0, 10, 7, 14, 8, 7],

        # set random seed
        random_seed=230,

        # loss
        sim_loss=[('ncc', {}, args.simE)],
        seg_loss=[('dice', {'n_class': n_classes, 'weight_type': 'Uniform', 'no_bg': False, 'eps': 1e-6}, args.segE)],
        reg_loss=[('gradient', {'norm': 'L2'}, args.regE)],
        learning_rate=args.lr,
        lr_mode='multiStep',  # const/plateau/...
        milestones=[0.9, 1],  # relative milestones e.g. 1 means the last epoch, 0.5 means the median epoch

        ckpoint_dir='../experiments/ckpoints/{}'.format(args.data),
        resume_dir='',
        preload=args.preload,
        device="cuda:0",
        debug_mode=args.debug,
    )

    main_set = ('MMRR-21', 'NKI-RS-21')
    extra_set = ('HLN-12', 'NKI-TRT-12', 'OASIS-TRT-20')
    train_file = main_set[args.fold]
    test_file = main_set[1 - args.fold]

    train_lists = (train_file + '-flip',) + tuple(extra + '-flip' for extra in extra_set)

    config['training_list_file'] = tuple(
        os.path.join(args.data_root, "mindboggle/{}.txt".format(file)) for file in train_lists)
    config['validation_list_file'] = os.path.join(args.data_root, "mindboggle/{}-flip-valid.txt".format(test_file))
    config['testing_list_file'] = os.path.join(args.data_root, "mindboggle/{}-flip.txt".format(test_file))
    config['log_dir'] = './{}/{}'.format(args.log_root, config['data'])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    exp = RegistrationExperiment(config)
    if not args.test_only:
        exp.train()
    exp.test()


if __name__ == '__main__':
    main()
