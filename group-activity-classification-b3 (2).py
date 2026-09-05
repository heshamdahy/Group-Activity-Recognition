# Generated from: group-activity-classification-b3 (2).ipynb
# Converted at: 2026-09-05T17:59:55.852Z
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

group_labels={'r_winpoint':0,'l_winpoint':1,'r-pass':2,'l-pass':3,'r_set':4,'l_set':5,'r_spike':6,'l-spike':7}
player_labels={'falling':0,'standing':1,'jumping':2,'blocking':3,'digging':4,'waiting':5,'moving':6,'spiking':7,'setting':8}

train_videos_list=['1' ,'3', '6', '7', '10', '13', '15', '16', '18', '22', '23', '31', '32', '36', '38', '39', '40', '41', '42', '48', '50', '52', '53', '54']
validate_videos_list=['0', '2' ,'8', '12' ,'17' ,'19' ,'24', '26',' 27', '28' ,'30' ,'33', '46' ,'49',' 51']
test_videos_list=['4' ,'5','9', '11' ,'14', '20', '21', '25', '29' ,'34' ,'35', '37', '43' ,'44', '45', '47']

train_videos=[os.path.join(base,vid) for vid in videos if vid in train_videos_list]
validate_videos=[os.path.join(base,vid) for vid in videos if vid in validate_videos_list]


test_videos=[os.path.join(base,vid) for vid in videos if vid in test_videos_list]

train_annotations=[os.path.join(vid,'annotations.txt') for vid in train_videos ]
validate_annotations=[os.path.join(vid,'annotations.txt') for vid in validate_videos]

test_annotations=[os.path.join(vid,'annotations.txt') for vid in test_videos]

def track_crop(ann):
    
  croped_frames=[]  
    
  with open(ann,'r') as f:
       
     path=ann.split('/')[:-1]
     path='/'.join(path)
     
     lines=f.readlines()
      
     
     for line in lines:
         
        
        frame=line.split()
        player_crops=frame[2:]
        image=frame[0]
        frame_id=image.split('.')[0]
        group_label=frame[1]
        group_label=group_labels[group_label]
        full_path=path+'/'+frame_id+'/'+image
        player_ann=[]
        crops=[]
        for idx , item in enumerate(player_crops):
             
             player_ann.append(item)
             if (idx+1)%5==0:
                x,y,w,h,player_label=player_ann
                player_label=player_labels[player_label]
                x,y,w,h=int(x),int(y),int(w),int(h)
                img=Image.open(full_path)
                crop_img=img.crop((x,y,x+w,y+h))
                crop=crop_img,player_label
                crops.append(crop)
                player_ann=[]
                 
        croped_frames.append((crops,group_label))
    

  return croped_frames
     


def track_data(annotations):
  crops=[]
  for ann in annotations:
     croped_img=track_crop(ann)
     for crop in croped_img:
           list_crops , group_label = crop
           crops.append((list_crops,group_label))
   
  return crops

train_data=track_data(train_annotations)
validate_data=track_data(validate_annotations)
test_data=track_data(test_annotations)

class volleyball(Dataset):
    def __init__(self,data,transforms=None):
        super().__init__()

        self.data=data
        self.transforms=transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        list_crops,group_label=self.data[idx]
        images=[]
        player_labels=[]
        for img , player_label in list_crops:
            
           
           if self.transforms:
               img=self.transforms(img)
           
           images.append(img) 
           player_labels.append(player_label)
         
        images=torch.stack(images)
        player_labels=torch.tensor(player_labels)
        
        return images,player_labels, group_label

preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset=volleyball(train_data,preprocess)
validate_dataset=volleyball(validate_data,preprocess)
test_dataset=volleyball(test_data,preprocess)
train_dataloader=DataLoader(train_dataset,1,shuffle=True)
validate_dataloader=DataLoader(validate_dataset,1,shuffle=True)
test_dataloader=DataLoader(test_dataset,1,shuffle=True)

criterion=nn.CrossEntropyLoss()

model = models.resnet50(pretrained=True).to(device)

model.fc=nn.Linear(2048,9).to(device)

for param in model.parameters():
    param.requires_grad=True

writer = SummaryWriter()

optimizer=optim.AdamW(model.parameters(),lr=0.0001)

def train_baseline_player_classifier(epochs,start_epoch):
  global model , optimizer ,train_dataloader,validate_dataloader,criterion ,device
  model.train()
  for epoch in range(start_epoch,epochs+start_epoch):
      total_loss,total,correct=3*(0,)
      for image , player_label,group_label in train_dataloader:
      
         image , player_label = image.to(device) , player_label.to(device)
         image=image.squeeze(0)
         player_label=player_label.squeeze(0)
         optimizer.zero_grad()
         output=model(image)
         loss=criterion(output,player_label)
         loss.backward()
         optimizer.step()
         total_loss+=loss.item()
         pred=torch.argmax(output,dim=1)
         correct+=(pred==player_label).sum().item()
         total+=player_label.size(0)

      
      
      print(f'epoch: {epoch+1} ----> loss: {total_loss/len(train_dataloader)} ----> Accuracy/train:{(100*correct)/total}   ')
      if (epoch+1)%5==0:
          torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'epoch':epoch},f'checkpoint_{epoch}.pth')
  
      
      writer.add_scalar('Loss/train',total_loss/len(train_dataloader) , epoch)
      writer.add_scalar('Accuracy/train',(100*correct)/total, epoch)

      model.eval()
      total,correct=2*(0,)
      with torch.no_grad():
             for image , player_label,group_label in validate_dataloader:
                 image , player_label = image.to(device) , player_label.to(device)
                 image=image.squeeze(0)
                 output=model(image)
                 pred=torch.argmax(output,dim=1)
                 correct+=(pred==player_label).sum().item()
                 total+=player_label.size(0)
          
      print(f'Accuracy/val: {(100*correct)/total}')
      writer.add_scalar('Accuracy/val',(100*correct)/total, epoch)
      model.train()
          
  return model

model=train_baseline_player_classifier(20,0)


children=list(model.children())[:-1]
feature_extractor=nn.Sequential(*children).to(device)

for param in feature_extractor.parameters():
    param.requires_grad=False

group_model=nn.Sequential(nn.Linear(2048,1024),nn.ReLU(),nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,8)).to(device)

group_optimizer=optim.AdamW(group_model.parameters(),lr=0.0001)

for epoch in range(20):
      total_loss,total,correct=3*(0,)
      for idx,(image,player_label,group_label) in enumerate(train_dataloader):
         
                 image,player_label,group_label=image.to(device),player_label.to(device),group_label.to(device)
                 image=image.squeeze(0)
                 player_label = player_label.squeeze(0)
                 group_label = group_label.squeeze(0)
                 group_optimizer.zero_grad()
                 output=feature_extractor(image)
                 output=torch.flatten(output,1)
                 output=torch.max(output,dim=0).values
                 group_image_pred=group_model(output)
            
                 
                 loss=criterion(group_image_pred,group_label)
                 loss.backward()
                 group_optimizer.step()
                 total_loss+=loss.item()
                 pred=torch.argmax(group_image_pred)
               
                 correct+=(pred==group_label).sum().item()
                 total+=1
               
     
      print(f'epoch: {epoch+1} ----> loss: {total_loss/len(train_dataloader)} ----> Accuracy/train:{(100*correct)/total}')
      if (epoch+1)%5==0:
          torch.save({'model_state_dict':group_model.state_dict(),'optimizer_state_dict':group_optimizer.state_dict(),'epoch':epoch},f'checkpoint_group_{epoch}.pth')
  
      
      writer.add_scalar('Loss/train',total_loss/len(train_dataloader),epoch)
      writer.add_scalar('Accuracy/train',(100*correct)/total, epoch)   
    
      group_model.eval()
      total,correct=2*(0,)
      with torch.no_grad():
             for image , player_label,group_label in validate_dataloader:
                 image,player_label,group_label=image.to(device),player_label.to(device),group_label.to(device)
                 image=image.squeeze(0)
                 player_label = player_label.squeeze(0)
                 group_label = group_label.squeeze(0)
                 output=feature_extractor(image)
                 output=torch.flatten(output,1)
                 output=torch.max(output,dim=0).values
                 group_image_pred=group_model(output)
                 pred=torch.argmax(group_image_pred)
                 correct+=(pred==group_label).sum().item()
                 total+=1
                 
      print(f'Accuracy/val: {(100*correct)/total}')
      writer.add_scalar('Accuracy/val',(100*correct)/total, epoch)
      group_model.train()