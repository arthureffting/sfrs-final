import os
import random

import torch
from shapely.geometry import Point
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter

from models.lol.lol_conv_lstm import MemoryLayer
from models.lol.lol_convolutions import PreLstmConvolutions, PostLstmConvolutions
from models.lol.patching.extract_tensor_patch import extract_tensor_patch
from models.lol.tsa import TemporalAttension


class LineOutlinerTsa(nn.Module):

    def __init__(self, path=None, patch_ratio=5, tsa_size=3, min_height=32, patch_size=64):
        super(LineOutlinerTsa, self).__init__()
        self.tsa_size = tsa_size
        self.min_height = min_height
        self.patch_size = patch_size
        self.patch_ratio = patch_ratio

        self.tsa = TemporalAttension(3)
        self.initial_convolutions = PreLstmConvolutions().cuda()
        self.memory_layer = MemoryLayer().cuda()
        self.final_convolutions = PostLstmConvolutions().cuda()

        self.fully_connected = nn.Linear(512, 5)
        self.fully_connected.bias.data[0] = 0
        self.fully_connected.bias.data[1] = 0
        self.fully_connected.bias.data[2] = 0
        self.fully_connected.bias.data[3] = 0
        self.fully_connected.bias.data[4] = 0

        # 0 -> angle to baseline
        # 1 -> next angle
        # 2 -> upper height scale
        # 3 -> lower height scale

        if path is not None and os.path.exists(path):
            state = torch.load(path, map_location=lambda storage, loc: storage)
            self.load_state_dict(state)
            self.eval()
        elif path is not None:
            print("\nCould not find path", path, "\n")

    def forward(self,
                img,
                sol_tensor,
                steps,
                reset_threshold=None,
                max_steps=None,
                disturb_sol=True,
                confidence_threshold=None,
                height_disturbance=0.5,
                angle_disturbance=30,
                translate_disturbance=10):

        desired_polygon_steps = torch.stack([sol_tensor.cuda()] + [p.cuda() for p in steps])

        img.cuda()

        # tensor([tsa, channels, width, height])
        input = ((255 / 128) - 1) * torch.ones((1, 3, self.patch_size, self.patch_size)).cuda()

        steps_ran = 0

        sol = {
            "upper_point": sol_tensor[0],
            "base_point": sol_tensor[1],
            "angle": sol_tensor[3][0],
        }

        if disturb_sol:
            x = random.uniform(0, translate_disturbance)
            y = random.uniform(0, translate_disturbance)
            sol["upper_point"][0] += x
            sol["upper_point"][1] += y
            sol["base_point"][0] += x
            sol["base_point"][1] += y
            sol["angle"] += random.uniform(-angle_disturbance, angle_disturbance)

        # current_height = current_height * (1 if not disturb_sol else (1 + random.uniform(0, height_disturbance)))

        current_height = Variable(torch.dist(sol["upper_point"].clone(), sol["base_point"].clone()),
                                  requires_grad=False).cuda()
        current_angle = Variable(sol["angle"].clone().cuda(), requires_grad=False)
        current_base = Variable(sol["base_point"].clone().cuda(), requires_grad=True)

        tsa_sequence = []
        results = []

        while max_steps is None or steps_ran < max_steps:

            torch.autograd.set_detect_anomaly(True)

            current_scale = (self.patch_ratio * current_height / self.patch_size).cuda()

            img_width = img.shape[2]
            img_height = img.shape[3]

            # current_angle = torch.mul(current_angle, -1)

            if img_width < current_base[0].item() or current_base[0].item() < 0:
                break
            if img_height < current_base[1].item() or current_base[0].item() < 0:
                break
            patch_parameters = torch.stack([current_base[0],  # x
                                            current_base[1],  # y
                                            torch.mul(torch.deg2rad(current_angle), -1),  # angle
                                            current_height]).unsqueeze(0)
            patch_parameters = patch_parameters.cuda()
            try:
                patch = extract_tensor_patch(img, patch_parameters, size=self.patch_size)  # size
            except:
                break

            # Shift input left
            # input = torch.stack([pic for pic in input[1:]] + [patch.squeeze(0)])

            input = torch.stack([pic for pic in input[-self.tsa_size:]] + [patch.squeeze(0)])

            y = input.cuda().unsqueeze(0)
            y = self.tsa(y)
            y = y[:, 1:, :, :, :]
            after_tsa_copy = y.detach().cpu().clone()
            tsa_sequence.append(after_tsa_copy)
            y = y.squeeze(0)
            y = self.initial_convolutions(y)
            y = y.unsqueeze(0)
            y = self.memory_layer(y)
            y = y.unsqueeze(0)
            y = self.final_convolutions(y)
            y = y.unsqueeze(0)
            y = torch.flatten(y, 1)
            y = torch.flatten(y, 0)
            y = self.fully_connected(y)

            size = input[-1, :, :, :].shape[1] / self.patch_ratio
            y[0] = torch.add(y[0], size) #x
            #y[1] #y
            #y[2] = torch.sigmoid(y[1]) # next angle
            y[3] = torch.add(torch.sigmoid(y[2]), 0.5)
            y[4] = torch.sigmoid(y[3])


            base_rotation_matrix = torch.stack(
                [torch.stack([torch.cos(torch.deg2rad(base_angle)), -1.0 * torch.sin(torch.deg2rad(base_angle))]),
                 torch.stack(
                     [1.0 * torch.sin(torch.deg2rad(base_angle)), torch.cos(torch.deg2rad(base_angle))])]).cuda()
            scale_matrix = torch.stack([torch.stack([current_scale, torch.tensor(0.).cuda()]),
                                        torch.stack([torch.tensor(0.).cuda(), current_scale])]).cuda()

            base_point = torch.stack([y[0], y[1]])
            base_point = torch.matmul(base_point, base_rotation_matrix.t())
            base_point = torch.matmul(base_point, scale_matrix)

            current_base = torch.add(current_base, base_point)
            current_height = torch.mul(current_height, y[3])
            current_angle = torch.add(base_angle, y[2])

            results.append(torch.stack([
                current_base,
                torch.stack([current_height, torch.mul(current_height, y[3])])
            ]))

            steps_ran += 1

            if reset_threshold is not None:
                base_as_point = Point(current_base[0].item(), current_base[1].item())
                gt_step = steps[steps_ran]
                gt_base_point = Point(gt_step[1][0].item(), gt_step[1][1].item())
                if base_as_point.distance(gt_base_point) > reset_threshold:
                    break

            if max_steps is None and steps_ran >= len(steps) - 1:
                break

        if steps_ran == 0:
            return None
        else:
            return torch.stack(results)
