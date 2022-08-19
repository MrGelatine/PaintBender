from paintBenderModel import *
import torch.optim as optim

model = PaintBender(core= vgg, target_path= PHOTO_PATH, style_path= STYLE_PATH, device= device, time_limit=5, snap_every=1000)
tar = model.create_target()

optimizer = optim.Adam([tar], lr=0.03)

model.start_time = time.time()
for ii in range(1, steps+1):
    los = model(tar)
    if not(los):
      break
    else:
      optimizer.zero_grad()
      los.backward()
      optimizer.step()
