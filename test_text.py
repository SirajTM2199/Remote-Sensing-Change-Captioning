import clip
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
model1,preprocess = clip.load('ViT-B/16',device='cuda:0')
from PIL import Image
import os
import concurrent.futures
from tqdm.auto import tqdm
import numpy as np
import json

def load_json(path):
    with open(path) as f:
        file = json.load(f)
    f.close()
    return file
def save_json(file,path):
    with open(path,'w') as f:
        json.dump(file,f)
    f.close()
    print("Saved Successfully")
def rem_print(word):
    t_word = word
    for _ in range(100 - len(t_word)):
        word = word + ' '
    print(word,end='\r')
    
# Text 

def Prompt_Maker(Initial_Image,Final_Image,model,preprocess,not_done=False,device='cpu'):
    Outputs = [["remote sensing image foreground objects"],
               ["remote sensing image background objects"],
               ["remote sensing image foreground objects"],
               ["remote sensing image background objects"]]
    
    Classes = ["Beach","Forest","Lake","Meadow","Mountain","Sea","Wetland","Cotton Field","Farmland","Prairie","Desert",
           "River","Tree","Shrubbery","Chaparral"," Fertile Land","Snow Land","Pond","Island Airport","Bridge","Freeway",
           "Harbor","Railway","Interchange","Intersection","Road","Highway","Basketball Court","Ground Track Field","Stadium",
           "TennisCourt","Golf Course","Dense Residential","Single-Family Residential","Building","Church","Cabin","Commercial Area",
           "Industrial Area","Oil Tank","Storage Tanks","Container","Mine Terrace","Campus","Park","Parking Lot","Square","Solar Panel",
           "Cars","Ship Airplane","Runway","Impermeable Surface"]
    
    TEXT= clip.tokenize(Classes).to(device)
    images = [Initial_Image,Final_Image]
   
    for i in range(len(images)):
        if not_done:
            IMAGE = Image.open(images[i])
            #print(type(IMAGE))
            IMAGE = preprocess(IMAGE).unsqueeze(0).to(device)
        else:
            IMAGE = images[i]
            
        with torch.no_grad():
            image_features,pos = model.encode_image(IMAGE)
            text_features = model.encode_text(TEXT)
            #print(IMAGE.shape)
            logits_per_image, logits_per_text = model(IMAGE,TEXT)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        #print(type(Outputs))   
        Outputs[2*i][0] += (",")
        for index in np.argsort(probs)[0][::-1][:9]:
            #results.append()
            Outputs[2*i][0] += Classes[index]
            Outputs[2*i][0] += ","
        Outputs[2*i][0] = Outputs[2*i][0][:-1]
    return Outputs


def make_texts(args):
    folder,device,model = args
    Directory = 'data/Levir-CC-dataset/images'
    Text_Dict = {}

    for image_name in tqdm(os.listdir(f'{Directory}/{folder}/A')):
        
        IMA = f'{Directory}/{folder}/A/{image_name}'
        IMB = f'{Directory}/{folder}/B/{image_name}'
        
        Text_Dict[image_name] = Prompt_Maker(IMA,IMB,model=model,preprocess=preprocess,not_done=True,device=device)
        #print('success!!')
    save_json(Text_Dict,f'text {folder} levircc.json')
    
make_texts(('test','cuda:0',model1))