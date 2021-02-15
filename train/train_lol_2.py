import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from shapely.geometry import Polygon, Point
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.lol import lol_dataset
from models.lol.lol_2 import LineOutlinerTsa
from models.lol.lol_dataset import LolDataset
from utils.dataset_parser import load_file_list_direct
from utils.files import create_folders, save_to_json
from utils.paint_lol_run import paint_model_run
from utils.wrapper import DatasetWrapper

parser = argparse.ArgumentParser(description='Prepare data for training')
parser.add_argument("--dataset", default="iam")
parser.add_argument("--batch_size", default=1)
parser.add_argument("--images_per_epoch", default=500)
parser.add_argument("--testing_images_per_epoch", default=20)
parser.add_argument("--stop_after_no_improvement", default=200)
parser.add_argument("--learning_rate", default=0.0002)

# Patching
parser.add_argument("--tsa_size", default=5)
parser.add_argument("--patch_ratio", default=5)
parser.add_argument("--patch_size", default=64)
parser.add_argument("--min_height", default=8)

# Training techniques
parser.add_argument("--name", default="training-8-height")
parser.add_argument("--reset-threshold", default=32)
parser.add_argument("--max_steps", default=6)
parser.add_argument("--random-sol", default=True)

parser.add_argument("--output", default="snapshots/lol")
args = parser.parse_args()

### SAVE ARGUMENTS
args_filename = os.path.join(args.output, args.name, 'args.json')
create_folders(args_filename)
with open(args_filename, 'w') as fp:
    json.dump(args.__dict__, fp, indent=4)

training_set_list_path = os.path.join("data", "iam", "prepared", "pages", "training.json")
training_set_list = load_file_list_direct(training_set_list_path)
train_dataset = LolDataset(training_set_list, augmentation=True)
train_dataloader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=lol_dataset.collate)
batches_per_epoch = int(int(args.images_per_epoch) / args.batch_size)
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

test_set_list_path = os.path.join("data", "iam", "prepared", "pages", "testing.json")
test_set_list = load_file_list_direct(test_set_list_path)
test_dataset = LolDataset(test_set_list)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=lol_dataset.collate)
test_dataloader = test_dataloader if args.testing_images_per_epoch is None else DatasetWrapper(test_dataloader,
                                                                                               int(
                                                                                                   args.testing_images_per_epoch))

validation_path = os.path.join("data", "iam", "prepared", "pages", "validation.json")
validation_list = load_file_list_direct(validation_path)
validation_set = LolDataset(validation_list[0:1])
validation_loader = DataLoader(validation_set,
                               batch_size=1,
                               shuffle=False,
                               num_workers=0,
                               collate_fn=lol_dataset.collate)

print("Loaded datasets")

lol = LineOutlinerTsa(tsa_size=args.tsa_size,
                      patch_size=args.patch_size,
                      min_height=args.min_height,
                      patch_ratio=args.patch_ratio)
lol.cuda()

optimizer = torch.optim.Adam(lol.parameters(), lr=float(args.learning_rate), weight_decay=0.01)

dtype = torch.cuda.FloatTensor

best_loss = np.inf
cnt_since_last_improvement = 0
all_epoch_data = []


def weighted_mse_loss(input, target, weight):
    squared_distance = (input - target) ** 2
    expand_weights = weight.expand_as(target)
    loss = squared_distance * expand_weights
    return torch.mean(loss)


def mixed_loss(sol, predicted_steps, desired_steps):
    base_height = Point(sol[0][0].item(), sol[0][1].item()).distance(Point(sol[2][0].item(), sol[2][1].item()))

    relative_upper_heights = []
    relative_lower_heights = []

    for tensor in desired_steps:
        upper_point = Point(tensor[0][0].item(), tensor[0][1].item())
        base_point = Point(tensor[1][0].item(), tensor[1][1].item())
        lower_point = Point(tensor[2][0].item(), tensor[2][1].item())
        upper_height = torch.tensor(base_point.distance(upper_point) / base_height, dtype=torch.float32).cuda()
        lower_height = torch.tensor(base_point.distance(lower_point) / base_height, dtype=torch.float32).cuda()
        relative_upper_heights.append(upper_height)
        relative_lower_heights.append(lower_height)

    baseline_loss = torch.nn.MSELoss()(torch.stack([torch.stack([p[0], p[1]]) for p in predicted_steps]),
                                       desired_steps[:, 1, [0, 1]].cuda())  # MSE of baseline
    upper_loss = torch.nn.BCELoss()(predicted_steps[:, 2], torch.stack(relative_upper_heights).cuda())
    lower_loss = torch.nn.BCELoss()(predicted_steps[:, 3], torch.stack(relative_lower_heights).cuda())
    confidence_loss = torch.nn.BCELoss()(predicted_steps[:, 4], desired_steps[:, 4, 0].cuda())
    #angle_loss = torch.nn.MSELoss()(predicted_steps[:, 4], desired_steps[:, 3, 0].cuda())

    return baseline_loss + upper_loss + lower_loss + confidence_loss


for epoch in range(1000):

    epoch_data = {
        "epoch": epoch,
    }

    print("[Epoch", epoch, "]")

    sum_loss = 0.0
    steps = 0.0
    total_steps_ran = 0
    lol.train()

    for index, x in enumerate(train_dataloader):
        # Only single batch for now
        x = x[0]
        if x is None:
            continue
        img = Variable(x['img'].type(dtype), requires_grad=False)[None, ...]
        ground_truth = x["steps"]

        sol_index = random.choice(range(len(ground_truth) - 4))

        sol = ground_truth[sol_index].cuda()
        predicted_steps = lol(img,
                              sol,
                              ground_truth[sol_index:],
                              min_steps=min(len(ground_truth[sol_index:]) - 1, args.tsa_size),
                              reset_threshold=args.reset_threshold,
                              disturb_sol=False)

        if predicted_steps is None:
            continue

        total_steps_ran += len(predicted_steps)

        if len(predicted_steps) == 0: break
        desired_steps = ground_truth[sol_index + 1: 1 + sol_index + len(predicted_steps)].cuda()

        desired_steps.cuda()
        # Transform [upper, base, lower] to [upper_height, base, lower_height]
        loss = mixed_loss(sol, predicted_steps, desired_steps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        steps += 1

        sys.stdout.write("\r[Training] " + str(1 + index) + "/" + str(len(train_dataloader)) + " | dice: " + str(
            round(sum_loss / steps, 3)) + " | " + "avg steps: " + str(round(total_steps_ran / steps, 3)))

    print()

    epoch_data["train"] = {
        "loss": sum_loss / steps,
        "avg_steps": total_steps_ran / steps,
    }

    sum_loss = 0.0
    steps = 0.0

    lol.eval()

    # Save epoch snapshot using some validation image
    model_path = os.path.join(args.output, args.name, 'last.pt')
    screenshot_path = os.path.join(args.output, args.name, "screenshots", str(epoch) + ".png")
    create_folders(screenshot_path)
    torch.save(lol.state_dict(), model_path)
    time.sleep(1)
    paint_model_run(model_path, validation_loader, destination=screenshot_path)

    for index, x in enumerate(test_dataloader):
        if x is None:
            continue
        x = x[0]
        img = Variable(x['img'].type(dtype), requires_grad=False)[None, ...]
        ground_truth = x["steps"]

        sol = ground_truth[0].cuda()
        predicted_steps = lol(img,
                              sol,
                              ground_truth,
                              max_steps=len(ground_truth) - 1,
                              disturb_sol=False)

        if predicted_steps is None:
            continue
        desired_steps = ground_truth[1:1 + len(predicted_steps), :, :]
        desired_steps.cuda()
        loss = mixed_loss(sol, predicted_steps, desired_steps)
        sum_loss += loss.item()
        steps += len(predicted_steps)
        sys.stdout.write(
            "\r[Testing] " + str(1 + index) + "/" + str(len(test_dataloader)) + " | dice: " + str(
                round(float(sum_loss / steps), 3)))

    cnt_since_last_improvement += 1

    epoch_data["test"] = {
        "loss": float(sum_loss / steps)
    }
    all_epoch_data.append(epoch_data)

    plt.plot(range(len(all_epoch_data)), [epoch["test"]["loss"] for epoch in all_epoch_data], label="Testing")
    plt.plot(range(len(all_epoch_data)), [epoch["train"]["loss"] for epoch in all_epoch_data], label="Training")
    plt.xlabel('Epoch')
    plt.ylabel('Inverse dice coefficient')
    plt.savefig(os.path.join(args.output, args.name, "plot.png"))
    loss_used = (sum_loss / steps)

    if loss_used < best_loss:
        cnt_since_last_improvement = 0
        best_loss = loss_used
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        torch.save(lol.state_dict(), os.path.join(args.output, args.name, 'best.pt'))
        print("\n[New best achieved]")
    else:
        print("\n[Current best]: ", round(best_loss, 3))

    epoch_json_path = os.path.join(args.output, args.name, "epochs", str(epoch) + ".json")
    create_folders(epoch_json_path)
    save_to_json(epoch_data, epoch_json_path)

    print()

    if cnt_since_last_improvement >= args.stop_after_no_improvement:
        break
