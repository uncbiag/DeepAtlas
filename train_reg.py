"""
Train a 3d unet
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
    parser.add_argument('--num_samples', '-ns', default=50, type=int,
                        help='number of samples for training')
    parser.add_argument('--num_epochs', '-ne', default=10, type=int,
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
    parser.add_argument('--data_root', '-root', default='/playpen-raid/zhenlinx/Data', type=str,
                        help='root of the data folder')
    parser.add_argument('--log_root', '-log', default='./logs', type=str,
                        help='root of the log folders that saves logs/checkpoints')
    args = parser.parse_args()

    n_classes = 5

    config = dict(
        debug_mode=args.debug,
        num_training_samples=100,
        resume_dir='',
        random_seed=123,
        data='OAI',
        n_epochs=args.num_epochs,
        samples_per_epoch=args.num_samples,
        batch_size=1,
        valid_batch_size=1,
        print_batch_period=50,
        valid_epoch_period=1,
        save_ckpts_epoch_period=1,

        model='voxel_morph_cvpr',   # UNet/UNet_light/UNet_light2
        model_settings=dict(input_channel=2,
                            output_channel=3,
                            enc_filters=[16, 32, 32, 32, 32],
                            dec_filters=[32, 32, 32, 8, 8]),

        n_classes=n_classes,
        class_name=[
            "FB",
            "FC",
            "TB",
            "TC"
        ],

        crop_size=[0, 20, 20],

        # config loss and optimizer
        sim_loss=[('ncc', {}, args.simE)],
        reg_loss=[('gradient', {'norm': 'L2'}, args.regE)],
        seg_loss=[('dice', {'n_class': 5, 'weight_type': 'Uniform', 'no_bg': False, 'eps': 1e-6}, args.segE)],
        learning_rate=1e-3,
        lr_mode='multiStep',  # const/plateau/...
        milestones=[0.9, 1],
        gamma=0.2
    )

    config.update(args.__dict__)

    config['data_dir'] = os.path.join(args.data_root, "OAI-ZIB/Nifti_resampled_rescaled_2Left_Affine2atlas")
    config['training_list_file'] = os.path.join(args.data_root, "OAI-ZIB/train.txt")
    config['validation_list_file'] = os.path.join(args.data_root, "OAI-ZIB/valid.txt")
    config['testing_list_file'] = os.path.join(args.data_root, "OAI-ZIB/test.txt")
    config['log_dir'] = './{}/{}'.format(args.log_root, config['data'])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    exp = RegistrationExperiment(config)
    if not args.test_only:
        exp.train()
    exp.test()


if __name__ == '__main__':
    main()
