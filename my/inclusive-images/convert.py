
# coding: utf-8

# # Data conversion for training the model

# In[25]:


import os
import shutil
import pandas as pd
import PIL.Image
from tqdm import tqdm


# In[20]:


DATA_PATH = "/mnt/kaggle/inclusive-images-challenge/"
TGT_PATH = "/mnt/kaggle-fast/inclusive-images-challenge/"


# In[21]:


os.makedirs(TGT_PATH, exist_ok=True)


# In[22]:


#df_label_names = pd.read_csv(f'{DATA_PATH}class-descriptions.csv')
print("Loading bounding box data...")
df_bboxes = pd.read_csv(f'{DATA_PATH}train_bounding_boxes.csv')


# In[26]:


print("Cleaning up old data...")
TRAIN_PATH = f'{TGT_PATH}train/'
if os.path.exists(TRAIN_PATH):
    shutil.rmtree(TRAIN_PATH)
os.makedirs(TRAIN_PATH)


# In[27]:


df = df_bboxes


# In[28]:


print("Converting %d bounding boxes" % len(df))
tgt_df = pd.DataFrame(columns=['ImageID', 'LabelName'])
prev_id = None
for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_id = row['ImageID']
    if img_id != prev_id:
        fname = f'{DATA_PATH}train/{img_id}.jpg'
        prev_id = img_id
        img = PIL.Image.open(fname)
        sub_idx = 0
        sub_map = {}
    h,w = img.size
    bbox = round(w*row['XMin'], 0), round(h*row['YMin'], 0), round(w*row['XMax'], 0), round(h*row['YMax'], 0)
    bbox = tuple(map(int, bbox))
    tgt_img_id = sub_map.get(bbox)
    if tgt_img_id is None:
        tgt_img_id = f'{img_id}_{sub_idx:05}'
        tgt_fname = f'{TRAIN_PATH}{tgt_img_id}.jpg'
        tgt_img = img.crop(bbox)
        tgt_img.save(tgt_fname)
        sub_map[bbox] = tgt_img_id
        sub_idx += 1
    tgt_label = row['LabelName']
    tgt_df = tgt_df.append({'ImageID': tgt_img_id, 'LabelName': tgt_label}, ignore_index=True)
tgt_df.to_csv(f'{TGT_PATH}train_proc.csv', index=False)

