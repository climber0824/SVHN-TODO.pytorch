import argparse
import os
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

from dataset import Dataset
from config import TestingConfig as Config
from model import Model


def _eval(path_to_checkpoint: str, path_to_data_dir: str, path_to_results_dir: str):
    os.makedirs(path_to_results_dir, exist_ok=True)

    # TODO: CODE BEGIN
    # raise NotImplementedError
    dataset = Dataset(path_to_data_dir, mode=Dataset.Mode.TEST)
    dataloader = DataLoader(dataset, batch_size=Config.Batch_Size, shuffle=False)

    model = Model()
    model.load(path_to_checkpoint)
    model.cuda()

    num_hits_for_digits = 0
    num_hits_for_length = 0
    # TODO: CODE END

    print(f'Start evaluating {path_to_checkpoint}')

    with torch.no_grad():
        # TODO: CODE BEGIN
        # raise NotImplementedError
        for batch_index, (images, length_labels, digits_labels) in tqdm.tqdm(enumerate(dataloader)):
            images, length_labels, digits_labels = (Variable(images.cuda()),
                                                    Variable(length_labels.cuda()),
                                                    [Variable(digit_labels.cuda()) for digit_labels in digits_labels])
            length_logits, digits_logits = model.eval().forward(images)

            # calculate accuracy of digits
            predictions = []
            for idx in range(5):
                predictions.append(digits_logits[idx].max(dim=1)[1])
                num_hits_for_digits += (np.array(predictions[idx], dtype='float32') == np.array(digits_labels[idx], dtype='float32')).sum().item()

            # calculate accuracy of length
            predictions = length_logits.max(dim=1)[1]
            num_hits_for_length += (predictions == length_labels).sum().item()

        # TODO: CODE END

        accuracy = num_hits_for_length / float(len(dataloader.dataset))
        print(f'Accuracy (length) = {accuracy:.4f}')
        accuracy = num_hits_for_digits / (5 * float(len(dataloader.dataset)))
        print(f'Accuracy (digits) = {accuracy:.4f}')

    with open(os.path.join(path_to_results_dir, 'accuracy.txt'), 'a') as fp:
        # step = path_to_checkpoint.split('-')[2]
        # step = step.split('.')[0]
        fp.write(f'{accuracy:.4f}\n')

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('checkpoint', type=str, help='path to evaluate checkpoint, e.g.: ./checkpoints/model-100.pth')
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-r', '--results_dir', default='./results', help='path to results directory')
        args = parser.parse_args()

        path_to_checkpoint = args.checkpoint
        path_to_data_dir = args.data_dir
        path_to_results_dir = args.results_dir
        _eval(path_to_checkpoint, path_to_data_dir, path_to_results_dir)

    main()
