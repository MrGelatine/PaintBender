## PaintBender
PyTorch Implementation of Neural Style Transfer from the paper "A Neural Algorithm of Artistic Style" (http://arxiv.org/abs/1508.06576)

## How to use
1. Download project and install all required вependencies
2. Put your style and to the appropriate folders(for example, dark_castle.jpg as target image and Starry_Night.jpg as style source)
3. Run this commmand in project folder:
```bash
python main.py eval --target dark_castle.jpg --style Starry_Night.jpg --time 5 --deep_style True
```
* `--target`: Name of image in target folder(for example,"my_picture.jpg").
* `--style`: Name of image in style folder(for example,"my_style.jpg").
* `--time`: How many minutes you can take model to create picture.
* `--deep_style`: If false, only change color pallet.
4. The finale version of picture will be located at results folder, but you also can see target snapshoots, that was had benn maiden during the transformation process
# Examples
Here is some examples of PaintBender work at deep style mode
<img src="https://github.com/MrGelatine/PaintBender/blob/main/results/chicago_new.png" />
<img src="https://github.com/MrGelatine/PaintBender/blob/main/results/dark_castle_new.png" />
<img src="https://github.com/MrGelatine/PaintBender/blob/main/results/human_son_new.png" />
<img src="https://github.com/MrGelatine/PaintBender/blob/main/results/mona_lisa_new.png" />


