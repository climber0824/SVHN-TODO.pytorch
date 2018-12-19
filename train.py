import argparse
import os
import time

from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from collections import deque

from config import TrainingConfig as Config
from model import Model
from dataset import Dataset


def _adjust_learning_rate(optimizer, step, initial_lr, decay_steps, decay_rate):
    lr = initial_lr * (decay_rate ** (step // decay_steps))
    for para_group in optimizer.param_groups:
        para_group['lr'] = lr
    return lr


def _train(path_to_data_dir: str, path_to_checkpoints_dir: str):
    os.makedirs(path_to_checkpoints_dir, exist_ok=True)

    # TODO: CODE BEGIN
    # raise NotImplementedError
    batch_size = Config.Batch_Size
    initial_learning_rate = Config.Learning_Rate
    steps_to_show_loss = Config.StepsToCheckLoss
    steps_to_save_model = Config.StepsToSnapshot
    steps_to_decay = Config.StepsToDecay
    decay_rate = Config.DecayRate
    step_to_terminate = Config.StepsToFinish

    train_loader = DataLoader(Dataset(path_to_data_dir, Dataset.Mode.TRAIN),
                              batch_size=batch_size, shuffle=True)

    model = Model()
    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=0.9, weight_decay=0.0005)

    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    should_stop = False

    # TODO: CODE END
    print('Start training')

    while not should_stop:
        # TODO: CODE BEGIN
        # raise NotImplementedError
        for batch_idx, (images, length_labels, digits_labels) in enumerate(train_loader):
            images, length_labels, digits_labels = (Variable(images.cuda()),
                                                    Variable(length_labels.cuda()),
                                                    [Variable(digit_labels.cuda()) for digit_labels in digits_labels])
            length_logits, digits_logits = model.train().forward(images)
            loss = model.loss(length_logits, digits_logits, length_labels, digits_labels)

            learning_rate = _adjust_learning_rate(optimizer, step, initial_learning_rate, steps_to_decay, decay_rate)

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

            if step % steps_to_save_model == 0:
                path_to_checkpoint = model.save(path_to_checkpoints_dir, step)
                print(f'Model saved to {path_to_checkpoint}')

            if step % step_to_terminate == 0:
                should_stop = True

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
