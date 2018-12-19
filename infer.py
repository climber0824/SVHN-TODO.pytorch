import argparse

import os
import torch
from PIL import Image

from dataset import Dataset
from model import Model


def _infer(path_to_image: str, path_to_checkpoint: str, path_to_results_dir: str):
    image = Image.open(path_to_image)
    image = image.resize((74, 74), Image.ANTIALIAS)
    image = Dataset.preprocess(image)

    model = Model().cuda()
    model.load(path_to_checkpoint)

    print('Start inferring')

    with torch.no_grad():
        images = image.unsqueeze(dim=0).cuda()

        # TODO: CODE BEGIN
        # raise NotImplementedError

        length_logits, digits_logits = model.eval().forward(images)

        # predict length
        pred_length = length_logits.max(dim=1)[1].item()
        print(f'length prediction =', pred_length)

        # predict digits
        pred_digits = ''
        for idx in range(5):
            if str(digits_logits[idx].max(dim=1)[1].item()) != '10':
                pred_digits = pred_digits + str(digits_logits[idx].max(dim=1)[1].item())
        # TODO: CODE END

        print(f'digits prediction = {pred_digits:s}')

    filename = path_to_image.split('/')[-1]  # would be xxx.png
    with open(os.path.join(path_to_results_dir, f'{filename}-prediction.txt'), 'w') as fp:
        fp.write(f'{pred_digits:s}')

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('image', type=str, help='path to image')
        parser.add_argument('-c', '--checkpoint', help='path to checkpoint')
        parser.add_argument('-r', '--results', help='path to results')
        args = parser.parse_args()

        path_to_image = args.image
        path_to_checkpoint = args.checkpoint
        path_to_results_dir = args.results
        _infer(path_to_image, path_to_checkpoint, path_to_results_dir)

    main()
