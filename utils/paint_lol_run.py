import torch
from shapely.geometry import Point, LineString

from models.lol.lol_2 import LineOutlinerTsa
from utils.files import create_folders
from utils.geometry import angle_between_points, get_new_point
from utils.painter import Painter


def paint_model_run(model_path, dataloader, destination="screenshots/run.png"):
    dtype = torch.cuda.FloatTensor

    img_path = None
    painter = None

    lol = LineOutlinerTsa(path=model_path)
    lol.cuda()

    for index, x in enumerate(dataloader):
        x = x[0]

        if img_path is None:
            painter = Painter(path=x["img_path"])
            img_path = x["img_path"]

        belongs = img_path == x["img_path"]

        if not belongs:
            continue

        img = x['img'].type(dtype)[None, ...]
        ground_truth = x["steps"]

        sol = ground_truth[0]
        base_height = Point(sol[0][0].item(), sol[0][1].item()).distance(Point(sol[2][0].item(), sol[2][1].item()))

        predicted_steps = lol(img, sol, ground_truth, max_steps=30, disturb_sol=False)

        # ground_truth_upper_steps = [Point(step[0][0].item(), step[0][1].item()) for step in predicted_steps]
        ground_truth_baseline_steps = [Point(step[0].item(), step[1].item()) for step in predicted_steps]

        upper_points = []
        lower_points = []

        for index, step in enumerate(ground_truth_baseline_steps[:-1]):
            upper_height, lower_height = base_height * predicted_steps[index][2].item(), \
                                         base_height * predicted_steps[index][3].item()
            next_step = ground_truth_baseline_steps[index + 1]
            angle_between_them = angle_between_points(step, next_step)
            upper_point = get_new_point(step, angle_between_them - 90, upper_height)
            lower_point = get_new_point(step, angle_between_them + 90, lower_height)
            painter.draw_line([upper_point, lower_point], line_width=1, color=(0, 0, 0, 0.5))
            upper_points.append(upper_point)
            lower_points.append(lower_point)

        for i in range(len(upper_points) - 1):
            confidence = predicted_steps[i, 4].item()
            painter.draw_area([upper_points[i], upper_points[i + 1], lower_points[i + 1], lower_points[i]],
                              fill_color=(confidence, 1 - confidence, 0, 0.05 + confidence))
        painter.draw_line(upper_points, line_width=2, color=(1, 0, 1, 1))
        painter.draw_line(lower_points, line_width=2, color=(1, 0, 1, 1))
        painter.draw_line(ground_truth_baseline_steps, line_width=2, color=(0, 0, 0, 0.5))

        sol = {
            "upper_point": ground_truth[0][0],
            "base_point": ground_truth[0][1],
            "angle": ground_truth[0][3][0],
        }

        sol_upper = Point(sol["upper_point"][0].item(), sol["upper_point"][1].item())
        sol_lower = Point(sol["base_point"][0].item(), sol["base_point"][1].item())

        painter.draw_line([sol_lower, sol_upper], color=(0, 1, 0, 1), line_width=4)
        painter.draw_point(sol_lower, color=(0, 1, 0, 1), radius=4)
        painter.draw_point(sol_upper, color=(0, 1, 0, 1), radius=4)

    create_folders(destination)
    painter.save(destination)
