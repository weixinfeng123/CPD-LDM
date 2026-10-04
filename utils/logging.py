import torch
import shutil
import os
import torchvision.utils as tvu


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)



# 模型恢复：可以通过保存的检查点恢复训练（例如加载 model_state_dict 和 optimizer_state_dict）并继续训练。
# 模型推理：保存的检查点可以在推理时加载，从而得到模型的预测结果。
def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')

# 加载保存的模型检查点。这个方法会从指定路径加载模型的状态，并根据需要将模型加载到指定的设备上
def load_checkpoint(path, device):
    if device is None:
        return torch.load(path)
    else:
        return torch.load(path, map_location=device)
