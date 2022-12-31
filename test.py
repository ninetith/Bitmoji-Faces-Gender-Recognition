import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import pandas as pd
testimg_path='../input/bitmoji-faces-gender-recognition/BitmojiDataset/testimages'
submission_path='../input/bitmoji-faces-gender-recognition/sample_submission.csv'

cnt=0
test_df=pd.read_csv(submission_path)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])
model = torch.load('./log.pth')
with torch.no_grad():
    parent_list=os.listdir(testimg_path)
    parent_list.sort()
    for img_name in parent_list:
        img_path = os.path.join(testimg_path, img_name)
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        if torch.cuda.is_available():
            img=img.cuda()
        img = img.unsqueeze(0)
        prect = int(model(img).argmax(1))
        if prect==0:
            prect=-1
        test_df['is_male'].iloc[cnt:cnt+1]=str(prect)
        cnt+=1
