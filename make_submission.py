from piecharts.utils.nms_utils import non_maximum_supression

from pathlib import Path
from piecharts.data.dataset import PiechartDataset
from piecharts.nn.models.config import TrainConfig
from piecharts.nn.models.smp_models import PSMModel

from piecharts.utils.area_utils import get_sectors
from piecharts.utils.image import pad_tensor_to_divisible_by_N
import pandas as pd
from tqdm import tqdm
import torch

def main():

    device = 'cuda:3'
    dir = Path('./models') / 'model_arch-unet_encoder_resnet34_freeze-False_center-False_loss-crossentropy/'

    config_path = dir / "config.json"

    with config_path.open("r", encoding="utf8") as f:
        json_config = f.read()
    # Parse the JSON string back into a Configuration instance
    config = TrainConfig.parse_raw(json_config)


    model = PSMModel(config.model, True).to(device)
    state_dict = torch.load(dir / 'epoch22.h5')
    model.load_state_dict(state_dict)
    dataset = PiechartDataset(Path("./data") / "raw", "val_and_test", "val")

    df = pd.DataFrame(columns=["id", "predicted_percentages"])

    with torch.no_grad():
        for i,(x,_) in tqdm(enumerate(dataset), total = len(dataset)):
            id = int(dataset.dataframe.iloc[i].filename[6:-4])

            x = x[None].to(device)
            x = pad_tensor_to_divisible_by_N(x, 32)

            y_pred = model(x)
            y_softmax = torch.nn.functional.softmax(y_pred, dim=1)

            arc_points = non_maximum_supression(y_softmax[0,1].cpu().numpy(), 0.5, 2)
            centers = non_maximum_supression(y_softmax[0,2].cpu().numpy(), 0.5, 2)
            
            all_distances = []

            for arc_point in arc_points:
                for center_point in centers:
                    all_distances.append((sum((arc_point - center_point)**2))**(1/2))

            if len(all_distances) != 0:
                min_val = int(min(all_distances))
                threshold = min_val*1.5

                centers_cropped = []
                for center in centers:
                    if int(center[0]) not in range(min_val, x.shape[-2] - min_val):
                        continue
                    if int(center[1]) not in range(min_val, x.shape[-1] - min_val):
                        continue
                    centers_cropped.append(center)

                arcs_ok = [False]*len(arc_points)
                centers_ok = [False] * len(centers_cropped)
                for i, arc_point in enumerate(arc_points):
                    for j, center_point in enumerate(centers_cropped):
                        dist = int(sum((arc_point - center_point)**2)**(1/2))
                        if dist < threshold:
                            arcs_ok[i] = True
                            centers_ok[j] = True

                arcs_clean = [point for point, ok in zip(arc_points, arcs_ok) if ok]
                centers_clean = [point for point, ok in zip(centers, centers_ok) if ok]
            else:
                arcs_clean = arc_points
                centers_clean = centers

            if len(centers_clean) == 0 or len(arcs_clean) == 0:
                secs = []
            else:
                secs = get_sectors([(x[1], x[0]) for x in centers_clean], [(x[1], x[0]) for x in arcs_clean], x.shape[-2])
            df.loc[len(df)] = {'id': id, 'predicted_percentages': secs}

            submission_df = pd.DataFrame({
                'id': df.id,
                'predicted_percentages': [str(lst) for lst in df.predicted_percentages]
            })

            submission_df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    main()

