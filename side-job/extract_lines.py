import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from shapely.geometry import Point
from torch.autograd import Variable

from models.lf.line_follower import LineFollower
from models.lol.patching import transformation_utils
from models.sol.sol import StartOfLineFinder
from utils import safe_load
from utils.files import create_folders, save_to_json

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract lines for Vincent :)')
    parser.add_argument("--input_height", default=64)
    parser.add_argument("--models_folder", default="side-job/models")
    parser.add_argument("--base_folder", default=".")
    parser.add_argument("--output", default="side-job/output")
    args = parser.parse_args()

    image_path_directories = ["side-job/scriptnet", "side-job/icdar"]
    image_paths = []
    skipped_images = {}

    too_big = []
    missing_images = ['2662-3-IMG_MAX_821515', '2662-3-IMG_MAX_821514',
                      '2662-3-IMG_MAX_821513', '1728-IMG_MAX_273335',
                      '1982-IMG_MAX_311268', '2878-IMG_MAX_881621',
                      '2967-IMG_MAX_924689',     ]

    for image_path_directory in image_path_directories:
        for root, folder, files in os.walk(image_path_directory):
            for f in files:
                if f.lower().endswith(".jpg") or f.lower().endswith(".png"):
                    stem = f.split(".")[0]
                    if stem in missing_images:
                        image_paths.append(os.path.join(args.base_folder, root, f))

    # Load Models
    print(image_paths)

    # SOL
    sol = StartOfLineFinder()
    sol_state = safe_load.torch_state(os.path.join(args.base_folder, args.models_folder, "sol.pt"))
    sol.load_state_dict(sol_state)
    sol.cuda()

    lf = LineFollower(args.input_height)
    lf_state = safe_load.torch_state(os.path.join(args.base_folder, args.models_folder, "lf.pt"))

    # special case for backward support of
    # previous way to save the LF weights
    if 'cnn' in lf_state:
        new_state = {}
        for k, v in lf_state.items():
            if k == 'cnn':
                for k2, v2 in v.items():
                    new_state[k + "." + k2] = v2
            if k == 'position_linear':
                for k2, v2 in v.state_dict().items():
                    new_state[k + "." + k2] = v2
            if k == 'learned_window':
                print("learned window found")
                # new_state[k] = torch.nn.Parameter(v.data)
        lf_state = new_state

    lf.load_state_dict(lf_state)
    lf.cuda()

    last_existent = None

    for image_path in sorted(image_paths):
        path_obj = Path(image_path)
        stem = path_obj.stem
        first_split = os.path.split(image_path)
        double_split = first_split[0].split("/")
        dataset = double_split[-1]
        this_image_folder = os.path.join(args.base_folder, "side-job", "output", dataset, stem, "lines")
        if os.path.exists(this_image_folder):
            last_existent = image_path

    for image_path in sorted(image_paths):

        print("[Processing]", image_path)
        path_obj = Path(image_path)
        stem = path_obj.stem
        first_split = os.path.split(image_path)
        double_split = first_split[0].split("/")
        dataset = double_split[-1]
        this_image_folder = os.path.join(args.base_folder, "side-job", "output", dataset, stem, "lines")

        #if os.path.exists(this_image_folder) and last_existent != image_path:
        #    print("[Skipping] ", image_path)
        #    continue

        org_img = cv2.imread(image_path)
        cv2.imwrite("test/" + Path(image_path).stem + ".png", org_img)
        org_img = cv2.imread("test/" + Path(image_path).stem + ".png")

        target_dim1 = 512
        s = target_dim1 / float(org_img.shape[1])

        pad_amount = 128
        org_img = np.pad(org_img, ((pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)), 'constant',
                         constant_values=255)
        before_padding = org_img

        target_dim0 = int(org_img.shape[0] * s)
        target_dim1 = int(org_img.shape[1] * s)

        full_img = org_img.astype(np.float32)
        full_img = full_img.transpose([2, 1, 0])[None, ...]
        full_img = torch.from_numpy(full_img)
        full_img = full_img / 128 - 1

        img = cv2.resize(org_img, (target_dim1, target_dim0), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)
        img = img.transpose([2, 1, 0])[None, ...]
        img = torch.from_numpy(img)
        img = img / 128 - 1

        # Parameters of E2E removed from there
        # because HW is not included here
        x = {
            "resized_img": img,
            "full_img": full_img,
            "resize_scale": 1.0 / s
        }
        use_full_img = False
        accpet_threshold = 0.1

        ##### E2E STUFF #######
        dtype = torch.cuda.FloatTensor
        sol_img = Variable(x['resized_img'].type(dtype), requires_grad=False, volatile=True)

        if use_full_img:
            img = Variable(x['full_img'].type(dtype), requires_grad=False, volatile=True)
            scale = x['resize_scale']
            results_scale = 1.0
        else:
            img = sol_img
            scale = 1.0
            results_scale = x['resize_scale']

        original_starts = sol(sol_img)

        predictions = transformation_utils.pt_xyrs_2_xyxy(original_starts)
        valid = []
        heights = []

        for prediction_index, prediction in enumerate(predictions[0, :, :]):
            is_valid = all([p > 0 for p in prediction[[1, 2, 3, 4]]])
            if is_valid:
                [x0, y0, x1, y1] = [prediction[i].item() for i in [1, 2, 3, 4]]
                p0 = Point(x0, y0)
                p1 = Point(x1, y1)
                height = p0.distance(p1)
                if height < 120:
                    heights.append(height)
                    valid.append(prediction_index)

        original_starts = original_starts[:, valid, :]
        start = original_starts

        # Take at least one point
        sorted_start, sorted_indices = torch.sort(start[..., 0:1], dim=1, descending=True)
        min_threshold = sorted_start[0, 1, 0].item()
        accpet_threshold = min(accpet_threshold, min_threshold)

        select = original_starts[..., 0:1] >= accpet_threshold
        select_idx = np.where(select.data.cpu().numpy())[1]

        select = select.expand(select.size(0), select.size(1), start.size(2))
        select = select.detach()
        start = start[select].view(start.size(0), -1, start.size(2))

        perform_forward = len(start.size()) == 3

        if not perform_forward:
            continue

        forward_img = img

        start = start.transpose(0, 1)

        positions = torch.cat([
            start[..., 1:3] * scale,
            start[..., 3:4],
            start[..., 4:5] * scale,
            start[..., 0:1]
        ], 2)

        p_interval = positions.size(0)
        lf_xy_positions = None
        line_imgs = []
        for p in range(0, min(positions.size(0), np.inf), p_interval):
            sub_positions = positions[p:p + p_interval, 0, :]
            sub_select_idx = select_idx[p:p + p_interval]

            batch_size = sub_positions.size(0)
            sub_positions = [sub_positions]

            expand_img = forward_img.expand(sub_positions[0].size(0), img.size(1), img.size(2), img.size(3))

            step_size = 5
            extra_bw = 1
            forward_steps = 40

            grid_line, _, out_positions, xy_positions = lf(expand_img, sub_positions, steps=step_size)
            grid_line, _, out_positions, xy_positions = lf(expand_img, [out_positions[step_size]],
                                                           steps=step_size + extra_bw, negate_lw=True)
            grid_line, _, out_positions, xy_positions = lf(expand_img, [out_positions[step_size + extra_bw]],
                                                           steps=forward_steps, allow_end_early=True)

            if lf_xy_positions is None:
                lf_xy_positions = xy_positions
            else:
                for i in range(len(lf_xy_positions)):
                    lf_xy_positions[i] = torch.cat([
                        lf_xy_positions[i],
                        xy_positions[i]
                    ])
            expand_img = expand_img.transpose(2, 3)

            hw_interval = p_interval
            for h in range(0, min(grid_line.size(0), np.inf), hw_interval):
                sub_out_positions = [o[h:h + hw_interval] for o in out_positions]
                sub_xy_positions = [o[h:h + hw_interval] for o in xy_positions]
                sub_sub_select_idx = sub_select_idx[h:h + hw_interval]

                line = torch.nn.functional.grid_sample(expand_img[h:h + hw_interval].detach(),
                                                       grid_line[h:h + hw_interval])
                line = line.transpose(2, 3)

                for l in line:
                    l = l.transpose(0, 1).transpose(1, 2)
                    l = (l + 1) * 128
                    l_np = l.data.cpu().numpy()
                    line_imgs.append(l_np)
                #     cv2.imwrite("example_line_out.png", l_np)
                #     print "Saved!"
                #     raw_input()

        for line_index, line_img in enumerate(line_imgs):

            # Remove empty columns
            columns_to_remove = []

            for column_index in range(line_img.shape[1]):
                column = line_img[:, column_index, :]
                sameColorColumn = np.all(column == column[0])
                if sameColorColumn: columns_to_remove.append(column_index)

            used_columns = [i for i in range(line_img.shape[1]) if i not in columns_to_remove]

            try:
                line_img = line_img[:, used_columns, :]
                line_img_path = os.path.join(this_image_folder, str(line_index) + ".png")
                create_folders(line_img_path)
                print("saving to", line_img_path)
                if line_img.shape[0] != 0 and line_img.shape[1] != 0:
                    cv2.imwrite(line_img_path, line_img)
            except:
                print("[Failed to save line] image: " + stem + " | line: " + str(line_index))

    save_to_json(skipped_images, os.path.join(args.base_folder, "side-job", "output", "skipped.json"))
