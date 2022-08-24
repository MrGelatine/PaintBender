import paintBenderModel
from torchvision import transforms, models
import torch.optim as optim
import torch
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaintBender")
    parser.add_argument('--time', '-tm', type=int, required=True, help='How many minutes you can take model to create picture')
    parser.add_argument('--target', '-trg', type=str, required=True, help='Name of image in target folder(for example,"my_picture.jpg")')
    parser.add_argument('--style', '-s', type=str, required=True, help='Name of image in style folder(for example,"my_style.jpg")')
    parser.add_argument('--deep_style', '-dp', type=bool, required=True, help='If false, only change color pallet')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PHOTO_PATH = r"./Mona_Lisa.jpg"
    STYLE_PATH = r"./Sky.jpg"
    kernel = models.vgg19(pretrained=True).features
    model = paintBenderModel.PaintBender(core= kernel, target_path= PHOTO_PATH, style_path= STYLE_PATH,
                                         device= device, time_limit=1, snap_every=1000)
    tar = model.create_target()

    optimizer = optim.Adam([tar], lr=0.03)

    steps = 50000

    model.start_time = time.time()
    while(True):
        los = model(tar)
        if not(los):
          break
        else:
          optimizer.zero_grad()
          los.backward()
          optimizer.step()
