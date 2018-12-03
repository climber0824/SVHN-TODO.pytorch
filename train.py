import argparse
import os
import time
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from collections import deque

from config import TrainingConfig
from model import Model
from dataset import Dataset


def _train(path_to_data_dir: str, path_to_checkpoints_dir: str):
    os.makedirs(path_to_checkpoints_dir, exist_ok=True)

    # TODO: CODE BEGIN
    # raise NotImplementedError
    batch_size = TrainingConfig.Batch_Size
    learning_rate = TrainingConfig.Learning_Rate
    patience = TrainingConfig.Patience
    steps_to_show_loss = TrainingConfig.StepsToCheckLoss
    steps_to_check = TrainingConfig.StepsToValidate
    steps_to_decay = TrainingConfig.StepsToDecayLearningRate
    steps_to_save_model = TrainingConfig.StepsToSnapshot
    decay_rate = TrainingConfig.DecayRate

    train_loader = DataLoader(Dataset(path_to_data_dir, Dataset.Mode.TRAIN),
                              batch_size=batch_size, shuffle=True)

    model = Model()
    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    step = 0
    time_checkpoint = time.time()
    best_accuracy = 0.0
    losses = deque(maxlen=100)
    should_stop = False

    # TODO: CODE END
    print('Start training')

    while not should_stop:
        # TODO: CODE BEGIN
        # raise NotImplementedError
        for batch_idx, (images, length_labels, digits_labels) in enumerate(train_loader):
            # images = np.array(images)
            # images = np.expand_dims(np.dot(images, [0.2989, 0.5870, 0.1140]), axis=3).astype(np.float32)
            # images = torch.from_numpy(images)
            images, length_labels, digits_labels = (Variable(images.cuda()),
                                                    Variable(length_labels.cuda()),
                                                    [Variable(digit_labels.cuda()) for digit_labels in digits_labels])
            # length_logits, digits_logits = model.train().forward(images)
            length_logits, digit1_logit, digit2_logit, digit3_logit, digit4_logit, digit5_logit = model.train().forward(images)
            # loss = model.loss(length_logits, digits_logits, length_labels, digits_labels)
            loss = model.loss(length_logits, digit1_logit, digit2_logit, digit3_logit, digit4_logit, digit5_logit,
                              length_labels, digits_labels[0], digits_labels[1], digits_labels[2], digits_labels[3],
                              digits_labels[4])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            losses.append(loss)

            if step % steps_to_show_loss == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec = steps_to_show_loss / elapsed_time
                avg_loss = sum(losses) / len(losses)
                print(f'[Step {step}] Loss = {avg_loss:.6f}, learning_rate = {learning_rate} '
                      f'({steps_per_sec:.2f} steps/sec)')

            # if step % steps_to_decay == 0:
            #     learning_rate = learning_rate * (decay_rate ** (step // steps_to_decay))
            #     optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
            '''
            if step % steps_to_check != 0:
                # evaluate on validation set
                continue
                
            * evaluate on validation set
            * check accuracy
            * check patience
            '''

            if step % steps_to_save_model == 0:
                path_to_checkpoint = model.save(path_to_checkpoints_dir, step)
                print(f'Model saved to {path_to_checkpoint}')

        # TODO: CODE END

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-c', '--checkpoints_dir', default='./checkpoints', help='path to checkpoints directory')
        args = parser.parse_args()

        path_to_data_dir = args.data_dir
        path_to_checkpoints_dir = args.checkpoints_dir

        _train(path_to_data_dir, path_to_checkpoints_dir)

    main()
