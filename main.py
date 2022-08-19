import paintBenderModel
from torchvision import transforms, models
import torch.optim as optim
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PHOTO_PATH = r"./Mona_Lisa.jpg"
STYLE_PATH = r"./Sky.jpg"
kernel = models.vgg19(pretrained=True).features
model = paintBenderModel.PaintBender(core= kernel, target_path= PHOTO_PATH, style_path= STYLE_PATH, device= device, time_limit=1, snap_every=1000)
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
