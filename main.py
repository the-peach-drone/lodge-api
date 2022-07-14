from fastapi import FastAPI, File, UploadFile
from typing import List
import os
import base64

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp


global net
net = smp.FPN(
    encoder_name='timm-efficientnet-b0', in_channels=4,
    classes=1, activation="sigmoid", encoder_weights="noisy-student",
)
net.eval()

app = FastAPI()

@app.get("/")
def root():
    return { "Lodge Segmentation": "Lodge Segmentation" }

@app.post("/inference")
async def inference(files: List[UploadFile] = File(...)):
    UPLOAD_DIRECTORY = "./static/"
    texts = []
    for file in files:
        contents = await file.read()
        with open(os.path.join(UPLOAD_DIRECTORY, file.filename), "wb") as fp:
            fp.write(contents)
        img = cv2.imread(os.path.join(UPLOAD_DIRECTORY, file.filename))
        out = infer(img) > 0.5
        out = cv2.cvtColor(np.float32(out), cv2.COLOR_GRAY2BGR)
        out = cv2.cvtColor(np.float32(out), cv2.COLOR_BGR2GRAY)
        print(out.max(), out.min())
        out = cv2.resize(out, dsize=(img.shape[1], img.shape[0])) * 255
        # cv2.imwrite("output.png", out*255)
        text = base64.b64encode(out)
        print(text)
        texts.append(text)
    return {"output": [text for text in texts]} 

def infer(img):
    items = preprocessing_1_4(img)
    print(items[0].size())
    mergei = []
    for item in items:
        img_r = net(item)
        img_p = np.transpose(img_r.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
        mergei.append(img_p)
    up = np.concatenate((mergei[0], mergei[1]), axis=1)
    down = np.concatenate((mergei[2], mergei[3]), axis=1)
    full = np.concatenate((up, down), axis=0)
    return full

def preprocessing_1_4(img):
    imgs = split_image(img)
    image_transform = transforms.Compose([transforms.ToTensor()])
    items = []
    for item in imgs:
        vi = get_vi(item, "ExGR")
        tmp = cv2.cvtColor(item, cv2.COLOR_BGR2BGRA)
        tmp[:, :, 3] = vi
        tmp = cv2.resize(tmp, dsize=(int(512), int(512)), interpolation=cv2.INTER_AREA)
        tmp = tmp/255.
        items.append(image_transform(tmp).float().unsqueeze(0))
    return items

def split_image(img):
    h, w, c = img.shape
    img_crop = [
        img[:int(int(h/2)), :int(w/2), :],
        img[int(h/2):, :int(w/2), :],
        img[:int(h/2), int(w/2):, :],
        img[int(h/2):, int(w/2):, :]
    ]
    return img_crop

def get_vi(img, vi=None):
    if vi == "ExG":
        return 2 * img[:, :, 1] - img[:, :, 0] - img[:, :, 2]
    elif vi == "ExR":
        return 1.4 * img[:, :, 0] - img[:, :, 1]
    elif vi == "ExGR":
        return 3 * img[:, :, 1] - 2.4 * img[:, :, 2] - img[:, :, 0]
    