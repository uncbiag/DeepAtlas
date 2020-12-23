"""
Train a 3d unet
"""
import os
import argparse

from models.segmentation import SegmentationExperiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-g', default='0', type=str,
                        help='index of used GPU')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='if debug mode')
    parser.add_argument('--preload', '-load', action='store_true',
                        help='if preload data into memory to speed up IO')
    parser.add_argument('--num-samples', '-ns', default=21, type=int,
                        help='number of samples for training')
    parser.add_argument('--num—epochs', '-ne', default=100, type=int,
                        help='number of samples for training')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--test_only', '-t', action='store_true',
                        help='only test model')
    parser.add_argument('--data-root', '-root', './data', type=str,
                        help='root of the data folder')
    args = parser.parse_args()

    n_classes = 32
    config = dict(
        debug_mode=args.debug,
        resume_dir='',
        random_seed=230,
        data='MindBoggle',
        n_epochs=args.num_epochs,
        samples_per_epoch=args.num_samples*2,  # due to flipping data augmentation
        batch_size=1,
        valid_batch_size=1,
        print_batch_period=50,
        valid_epoch_period=1,
        save_ckpts_epoch_period=1,

        model='UNet_light',
        model_settings={'in_channel': 1, 'n_classes': n_classes, 'bias': True, 'BN': True},
        n_classes=n_classes,
        class_name={k: str(k) for k in range(1, n_classes)},

        crop_size=[0, 10, 7, 14, 8, 7],

        # config loss and optimizer
        loss='dice',  # cross_entropy/dice/focal_loss/genDice
        loss_settings={'n_class': n_classes, 'weight_type': 'Uniform', 'no_bg': False, 'softmax': True, 'eps': 1e-6},

        learning_rate=1e-3,
        lr_mode='multiStep',  # const/plateau/...
        milestones=[0.5, 1],
        gamma=0.2
    )

    config.update(args.__dict__)

    train_set = ('MMRR-21', 'HLN-12', 'NKI-TRT-12', 'OASIS-TRT-20')
    test_set = 'NKI-RS-21'  # subset for validation and testing

    if config['num_samples'] == 21:
        # use only MMRR-21 for training
        train_lists = (train_file + '-flip' for train_file in train_set[0:1])
    elif config['num_samples'] == 65:
        train_lists = (train_file + '-flip' for train_file in train_set)
    else:
        raise ValueError("n_seg has to be 21 or 65 for mindboggle data but got {}".format(config['num_samples']))

    testing_list = "NKI-RS-21-train.txt"

    config['data_dir'] = os.path.join(args.data_root, "mindboggle")
    config['valid_data_dir'] = os.path.join(args.data_root, "mindboggle")
    config['training_list_file'] = tuple(os.path.join(args.data_root, "mindboggle/{}.txt".format(file))
                                         for file in train_lists)
    config['validation_list_file'] = os.path.join(args.data_root, "mindboggle/{}-valid.txt".format(test_set))
    config['testing_list_file'] = os.path.join(args.data_root, "mindboggle/{}".format(testing_list))
    config['log_dir'] = './results/{}'.format(config['data'])

    if not args.leaf:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    exp = SegmentationExperiment(config)
    if not args.test_only:
        exp.train()
    exp.test()


if __name__ == '__main__':
    main()
