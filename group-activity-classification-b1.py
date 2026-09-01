# Generated from: group-activity-classification-b1.ipynb
# Converted at: 2026-08-31T22:24:31.757Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader,Dataset
import  torchvision.transforms as transforms 
import torchvision.models as models
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


import os 

base='/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos/'
videos=os.listdir(base)


train_videos_list=['1' ,'3', '6', '7', '10', '13', '15', '16', '18', '22', '23', '31', '32', '36', '38', '39', '40', '41', '42', '48', '50', '52', '53', '54']
validate_videos_list=['0', '2' ,'8', '12' ,'17' ,'19' ,'24', '26',' 27', '28' ,'30' ,'33', '46' ,'49',' 51']
test_videos_list=['4' ,'5','9', '11' ,'14', '20', '21', '25', '29' ,'34' ,'35', '37', '43' ,'44', '45', '47']

train_videos=[os.path.join(base,vid) for vid in videos if vid in train_videos_list]
validate_videos=[os.path.join(base,vid) for vid in videos if vid in validate_videos_list]
test_videos=[os.path.join(base,vid) for vid in videos if vid in test_videos_list]

group_labels={'r_winpoint':0,'l_winpoint':1,'r-pass':2,'l-pass':3,'r_set':4,'l_set':5,'r_spike':6,'l-spike':7}

train_annotations=[os.path.join(vid,'annotations.txt') for vid in train_videos ]
validate_annotations=[os.path.join(vid,'annotations.txt') for vid in validate_videos]
test_annotations=[os.path.join(vid,'annotations.txt') for vid in test_videos]

def track_annotate_image_level(ann,video,idx):

  with open(ann,'r') as f:
     line=f.readlines()
    
     for _,li in enumerate(line):
   
        new_line=li.split(',')
        new_line=new_line[0].split()
       
        frame_id=new_line[0].split('.')[0]
        group_activity=new_line[1]
        if frame_id not in video[idx]:
            video[idx][frame_id]={}

        video[idx][frame_id]=group_activity
      
     return video 
       

train_annotation_mark=[7,22,50,23,10,36,41,39,32,42,52,38,31,53,18,16,13,15,3,1,40,6,54,48]
validate_annotation_mark=[17,19,2,8,12,49,0,28,26,30,46,33,24]
test_annotation_mark=[47,35,5,20,45,25,34,43,14,4,9,21,44,11,37,29]


def track_annotate_video(annotations,annotation_mark):
  videos_data=[]
  for idx,ann in enumerate(annotations):
    video={}
    mark_annotate=annotation_mark[idx]
    if mark_annotate not in video:
      video[mark_annotate]={}

    
    video = track_annotate_image_level(ann,video,mark_annotate)
    videos_data.append(video)
      
  return videos_data 

train_videos_data=track_annotate_video(train_annotations,train_annotation_mark)
validate_videos_data=track_annotate_video(validate_annotations,validate_annotation_mark)
test_videos_data=track_annotate_video(test_annotations,test_annotation_mark)


def track_video(videos_data):
  final_videos=[]
    
  for video in videos_data:
    for video_idx,frames in video.items():
       video_clip=[]
       for frame_id, label in frames.items():
    
          path=os.path.join(base,str(video_idx))
          path=os.path.join(path,frame_id)
         
          images=os.listdir(path)
        
          target_frame=frame_id+'.jpg'
       
          idx_target_frame=images.index(target_frame)
          
          if idx_target_frame ==0:
              images=images[1:10]
          elif idx_target_frame==len(images)-1:
                    images=images[-2:-12]
          else :
              idx_before_target=idx_target_frame-5
              idx_after_target=idx_target_frame+5
              if idx_before_target>=0 and idx_after_target<=len(images):
                images_before_target=images[idx_before_target:idx_target_frame-1]
                images_after_target=images[idx_target_frame+1:idx_after_target]
                images=images_before_target+images_after_target
              else :
                  if idx_before_target<0:
                     images_before_target=images[0:idx_target_frame-1]
                     idx_after_target=idx_target_frame+(5-idx_target_frame)+5
                     images_after_target=images[idx_target_frame+1:idx_after_target]
                     images=images_before_target+images_after_target
                      
                  elif idx_after_target>len(images):
                     images_after_target=images[idx_target_frame+1:]
                     idx_before_target=idx_target_frame-5-(idx_after_target-len(images))
                     images_before_target=images[idx_before_target:idx_target_frame-1]
                     images=images_before_target+images_after_target
                      
               
              images=[os.path.join(path,img) for img in images]
            
              label=group_labels[label]
            
              clip=(images,label,os.path.join(path,target_frame))
              
              video_clip.append(clip)
    
    final_videos.append(video_clip)
  return final_videos

final_train_videos=track_video(train_videos_data)
final_validate_videos=track_video(validate_videos_data)
final_test_videos=track_video(test_videos_data)


def track_data(final_videos):
  final_data=[]
  for video in final_videos:
    for clip in video:
          final_data.append(clip)

  final_dataset=[]
  for clip in final_data:
     frames,label,target=clip
     frames.append(target)
     frames=sorted(frames,key=lambda x : int(x.split('/')[-1].split('.')[0]))
     clip=(frames,label)
     final_dataset.append(clip)
   
  dataset=[]
  for clip in final_dataset:
    frames , label = clip
    for frame in frames :
        dataset.append((frame,label))
   
  return dataset

train_dataset=track_data(final_train_videos)
validate_dataset=track_data(final_validate_videos)
test_dataset=track_data(final_test_videos)

class volleyball(Dataset):
    def __init__(self,data,transforms=None):
        super().__init__()

        self.data=data
        self.transforms=transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img,label=self.data[idx]
        image=Image.open(img)

        if self.transforms:
            image=self.transforms(image)

        return image , label
    

preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vollyball_train_dataset=volleyball(train_dataset,preprocess)
vollyball_validate_dataset=volleyball(validate_dataset,preprocess)
vollyball_test_dataset=volleyball(test_dataset,preprocess)
train_dataloader=DataLoader(vollyball_train_dataset,32,shuffle=True)
validate_dataloader=DataLoader(vollyball_validate_dataset,32,shuffle=True)
test_dataloader=DataLoader(vollyball_test_dataset,32,shuffle=True)

criterion=nn.CrossEntropyLoss()

model = models.resnet50(pretrained=True).to(device)


model.fc=nn.Linear(2048,8).to(device)

for param in model.parameters():
    param.requires_grad=True

optimizer=optim.AdamW(model.parameters(),lr=0.0001)

def train_baseline_image_classifier(epochs,start_epoch):
  global model , optimizer ,train_dataloader,validate_dataloader,criterion ,device
  model.train()
  for epoch in range(start_epoch,epochs+start_epoch):
      total_loss,total,correct=3*(0,)
      for image , label in train_dataloader:
      
         image , label = image.to(device) , label.to(device)
         optimizer.zero_grad()
         output=model(image)
         loss=criterion(output,label)
         loss.backward()
         optimizer.step()
         total_loss+=loss.item()
         pred=torch.argmax(output,dim=1)
         correct+=(pred==label).sum().item()
         total+=label.size(0)

      
      
      print(f'epoch: {epoch+1} ----> loss: {total_loss/len(train_dataloader)} ----> Accuracy/train:{(100*correct)/total}   ')
      if (epoch+1)%20==0:
          torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'epoch':epoch},f'checkpoint_{epoch}.pth')
  
      if (epoch+1)%10==0:
          writer.add_scalar('Loss/train',total_loss/len(train_dataloader) , epoch)
          writer.add_scalar('Accuracy/train',(100*correct)/total, epoch)

          model.eval()
          total,correct=2*(0,)
          with torch.no_grad():
             for image , label in validate_dataloader:
                 image , label = image.to(device) , label.to(device)
                 output=model(image)
                 pred=torch.argmax(output,dim=1)
                 correct+=(pred==label).sum().item()
                 total+=label.size(0)
          
          print(f'Accuracy/val: {(100*correct)/total}')
          writer.add_scalar('Accuracy/val',(100*correct)/total, epoch)
          model.train()
          
  return model      

model=train_baseline_image_classifier(100,0)

model.eval()
with torch.no_grad(): 
    total,correct=2*(0,)
    for image , label in test_dataloader:
        image , label = image.to(device),label.to(device)
        output=model(image)
        pred=torch.argmax(output,dim=1)
        correct+=(pred==label).sum().item()
        total+=label.size(0)

    
print(f' Accuracy/test:{100*correct/total}')
writer.add_scalar('Accuracy/test',(100*correct)/total)
