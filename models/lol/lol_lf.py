import os
import random

import torch
from shapely.geometry import Point
from torch import nn
from torch.autograd import Variable

from models.lol.lol_baseline_module import BaselineModule
from models.lol.lol_conv_lstm import MemoryLayer
from models.lol.lol_convolutions import PreLstmConvolutions, PostLstmConvolutions
from models.lol.lol_outline_module import OutlineModule
from models.lol.patching.extract_tensor_patch import extract_tensor_patch
from models.lol.stop_module import StopModule
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
        self.baseline_module = BaselineModule()
        self.upper_module = OutlineModule()
        self.lower_module = OutlineModule()
        self.stop_module = StopModule()

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

        current_height = torch.dist(sol["upper_point"].clone(), sol["base_point"].clone()).cuda()
        current_height = current_height * (1 if not disturb_sol else (1 + random.uniform(0, height_disturbance)))

        current_angle = sol["angle"].clone().cuda()
        current_base = sol["base_point"].clone().cuda()

        tsa_sequence = []

        upper_points = []
        baseline_points = []
        lower_points = []
        stop_confidences = []
        angle_changes = [0.0]

        while (max_steps is None or steps_ran < max_steps):

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
            image_output = torch.flatten(y, 0)

            # Flattened array outcoming from CONV -> TSA -> CONV LSTM -> CONV

            angle_input = torch.tensor(angle_changes, dtype=torch.float32).cuda()

            size = input[0, :, :, :].shape[1] / self.patch_ratio
            baseline_output = self.baseline_module(image_output, angle_input)
            baseline_output[0] = torch.add(baseline_output[0], size)

            # y[2] = torch.add(y[2], size)
            # y[3] = torch.add(y[3], -size)
            # y[4] = torch.add(y[4], size)

            scale_matrix = torch.stack([torch.stack([current_scale, torch.tensor(0.).cuda()]),
                                        torch.stack([torch.tensor(0.).cuda(), current_scale])]).cuda()
            # Finds the next base point
            base_rotation_matrix = torch.stack(
                [torch.stack([torch.cos(torch.deg2rad(current_angle)), -1.0 * torch.sin(torch.deg2rad(current_angle))]),
                 torch.stack(
                     [1.0 * torch.sin(torch.deg2rad(current_angle)), torch.cos(torch.deg2rad(current_angle))])]).cuda()

            # Create a vector to represent the new base
            base_point = torch.stack([baseline_output[0], baseline_output[1]])
            base_point = torch.matmul(base_point, base_rotation_matrix.t())
            base_point = torch.matmul(base_point, scale_matrix)

            # After predicting the baseline
            # detach hidden layer and predict outline
            #
            # image_output.detach()
            #

            outline_input = Variable(image_output.clone().detach(), requires_grad=False)
            outline_reference = Variable(current_base.clone().detach(), requires_grad=False)
            angle_reference = Variable(torch.tensor(angle_changes, dtype=torch.float32).cuda(), requires_grad=False)
            upper_output = self.upper_module(outline_input, angle_reference)
            upper_output[1] = torch.add(upper_output[1], -size)
            upper_output[0] = torch.add(upper_output[0], size)
            upper_point = torch.stack([upper_output[0], upper_output[1]])
            upper_point = torch.matmul(upper_point, base_rotation_matrix.t())
            upper_point = torch.matmul(upper_point, scale_matrix)
            upper_point = torch.add(upper_point, outline_reference)
            upper_points.append(upper_point)

            lower_output = self.lower_module(outline_input, angle_reference)
            lower_output[0] = torch.add(lower_output[0], size)
            lower_point = torch.stack([lower_output[0], lower_output[1]])
            lower_point = torch.matmul(lower_point, base_rotation_matrix.t())
            lower_point = torch.matmul(lower_point, scale_matrix)
            lower_point = torch.add(lower_point, outline_reference)
            lower_points.append(lower_point)

            stop_output = self.stop_module(outline_input, angle_reference)

            # Update base and angle for patching
            angle_changes.append(baseline_output[2].item())
            current_angle = torch.add(current_angle, baseline_output[2])
            current_base = torch.add(current_base, base_point)
            baseline_points.append(current_base)
            stop_confidences.append(stop_output[0])

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
            return None, None, None, None
        else:
            return torch.stack(upper_points), \
                   torch.stack(baseline_points), \
                   torch.stack(lower_points), \
                   torch.stack(stop_confidences)
