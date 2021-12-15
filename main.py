import argparse
import train
import os
import ast

#Training Code

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=ast.literal_eval, default=True)
    parser.add_argument("--resume", type=ast.literal_eval, default=False)
    parser.add_argument("--noise", type=ast.literal_eval, default=False)
    parser.add_argument("--use_cuda", type=ast.literal_eval, default=True)
    parser.add_argument("--seed", type=int, default=20210615)
    parser.add_argument("--trainset", type=str, default="./train_image/")
    parser.add_argument("--testset", type=str, default="./test_image/")

    parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')
    parser.add_argument('--ckpt_path', default='./generator/checkpoint/', type=str,
                        metavar='PATH', help='path to checkpoints')
    # parser.add_argument('--fused_img_path', default='./generator/fused_result/', type=str,
    #                     metavar='PATH', help='path to save images')
    # parser.add_argument('--y_map_path', default='./generator/y_maps/', type=str,
    #                     metavar='PATH', help='path to save channel_y_maps')
    parser.add_argument('--test_fused_result', default='./generator/test_img/', type=str,
                        metavar='PATH', help='path to save test_fused_result')

    parser.add_argument("--max_epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight", type=float, default=[10,5,100])
    parser.add_argument("--decay_interval", type=int, default=120)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--epochs_per_eval", type=int, default=600)#
    parser.add_argument("--epochs_per_save", type=int, default=1)#
    parser.add_argument("--train_batch_size", type=int, default=16)#
    parser.add_argument("--test_batch_size", type=int, default=1)#

    return parser.parse_args()


def main(cfg):
    trainning = train.Trainer(cfg)
    if cfg.train:
        trainning.start()

if __name__ == "__main__":
    config = parse_config()
    print(config.use_cuda)
    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    print(config.train_batch_size)
    main(config)
