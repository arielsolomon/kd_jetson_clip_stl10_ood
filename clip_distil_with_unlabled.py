import argparse
from stl10_utils import (
    precompute_clip_stl10_train_image_embeddings,
    precompute_clip_stl10_unlabeled_image_embeddings,
    precompute_clip_stl10_test_image_embeddings,
    precompute_clip_stl10_text_embeddings,
    train_resnet18_zero_shot,
    train_resnet18_linear_probe
)

def f_name(f):
    return f.__name__
def run(exp_name, data_path):
    precompute_clip_stl10_train_image_embeddings(exp_name, data_path)
    precompute_clip_stl10_unlabeled_image_embeddings(exp_name, data_path)
    precompute_clip_stl10_test_image_embeddings(exp_name, data_path)
    precompute_clip_stl10_text_embeddings(exp_name)
    train_resnet18_zero_shot(f_name(train_resnet18_zero_shot), exp_name)
    train_resnet18_linear_probe(f_name(train_resnet18_linear_probe), exp_name)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/Data/federated_learning/kd_jetson_clip_stl10_ood/data/stl10', help='bin STL10 path')
    parser.add_argument('--exp-name', type=str, default='ood_gaussian_noise04_unlabeled', help='experiment name')
    opt = parser.parse_args()
    return opt
def main(opt):

    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)