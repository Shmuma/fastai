
# coding: utf-8

# # Data conversion for training the model

# In[32]:


import os
import shutil
import pandas as pd
import PIL.Image
from tqdm import tqdm
import concurrent.futures as futures


# In[20]:


DATA_PATH = "/mnt/kaggle/inclusive-images-challenge/"
TGT_PATH = "/mnt/kaggle-fast/inclusive-images-challenge/"


# In[21]:


os.makedirs(TGT_PATH, exist_ok=True)


# In[22]:


#df_label_names = pd.read_csv(f'{DATA_PATH}class-descriptions.csv')
print("Loading bounding box data...")
df_bboxes = pd.read_csv(f'{DATA_PATH}train_bounding_boxes.csv')


# In[42]:


TRAIN_PATH = f'{TGT_PATH}train/'
if not os.path.exists(TRAIN_PATH):
    os.makedirs(TRAIN_PATH)


# In[36]:


class Job:
    def __init__(self, img_id):
        self.img_id = img_id
        self.sub_map = {}
        self.sub_idx = 0
        self.images_labels = []
        
    def add_row(self, row):
        if self.img_id != row['ImageID']:
            return False
        label = row['LabelName']
        bbox = [min(1.0, max(0, round(row[t], 5))) for t in ('XMin', 'YMin', 'XMax', 'YMax')]
        bbox = tuple(bbox)
        tgt_img_id = self.sub_map.get(bbox)
        if tgt_img_id is None:
            tgt_img_id = f'{self.img_id}_{self.sub_idx:05}'
            self.sub_idx += 1
            self.sub_map[bbox] = tgt_img_id
        self.images_labels.append((tgt_img_id, label))
        return True

    def add_labels(self, df):
        for img, label in self.images_labels:
            df = df.append({'ImageID': img, 'LabelName': label}, ignore_index=True)
        return df
    
    def submit(self, executor):
        return executor.submit(do_job, self.img_id, self.sub_map)
        
def do_job(img_id, sub_map):
    processing_needed = False
    for bbox, tgt_img_id in sub_map.items():
        tgt_fname = f'{TRAIN_PATH}{tgt_img_id}.jpg'
        if os.path.exists(tgt_fname):
            continue
        processing_needed = True
    if not processing_needed:
        return
    
    fname = f'{DATA_PATH}train/{img_id}.jpg'
    img = PIL.Image.open(fname)
    w, h = img.size
    for bbox, tgt_img_id in sub_map.items():
        tgt_fname = f'{TRAIN_PATH}{tgt_img_id}.jpg'
        if os.path.exists(tgt_fname):
            continue
        crop = (w * bbox[0], h * bbox[1], w * bbox[2], h * bbox[3])
        crop = list(map(int, crop))
        tgt_img = img.crop(crop)
        tgt_img.save(tgt_fname)


# In[46]:


NUM_JOBS = 6
MAX_CONCURRENT_JOBS = 10000
WAIT_SECONDS = 10
#df = df_bboxes[:30000]
df = df_bboxes


# In[47]:


print("Converting %d bounding boxes" % len(df))
tgt_df = pd.DataFrame(columns=['ImageID', 'LabelName'])
fs = []
with futures.ThreadPoolExecutor(max_workers=NUM_JOBS) as executor:
    job = None
    try:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            img_id = row['ImageID']
            if job is None:
                job = Job(img_id)
            if not job.add_row(row):
                fs.append(job.submit(executor))
                tgt_df = job.add_labels(tgt_df)
                job = Job(img_id)
                job.add_row(row)
                if len(fs) >= MAX_CONCURRENT_JOBS:                
                    done_fs, fs = futures.wait(fs, timeout=WAIT_SECONDS)
                    fs = list(fs)
                    print("Collected %d completed jobs" % len(done_fs))
        fs.append(job.submit(executor))
        tgt_df = job.add_labels(tgt_df)
        tgt_df.to_csv(f'{TGT_PATH}train_proc.csv', index=False)
        print("Waiting for %d jobs to be completed" % len(fs))
        futures.wait(fs)
    except KeyboardInterrupt:
        print("Interrupt pressed, waiting for %d jobs to be completed gracefully" % len(fs))
        futures.wait(fs)

