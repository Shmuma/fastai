
# coding: utf-8

# # Data conversion for training the model

# In[1]:


import os
import shutil
import pandas as pd
import PIL.Image
from tqdm import tqdm
import concurrent.futures as futures


# In[3]:


DATA_PATH = "/mnt/stg/inclusive-images-challenge/"
RAW_PATH = f'{DATA_PATH}raw/'
TGT_PATH = f'{DATA_PATH}train/'


# In[4]:


os.makedirs(TGT_PATH, exist_ok=True)


# In[5]:


print("Loading labels data frame...")
df_label_names = pd.read_csv(f'{DATA_PATH}class-descriptions.csv')
df_trainable_labels = pd.read_csv(f'{DATA_PATH}classes-trainable.csv')
print("Loading bounding box data...")
df_bboxes = pd.read_csv(f'{DATA_PATH}train_bounding_boxes.csv')


# In[6]:


labels_set = set(df_trainable_labels.label_code.tolist())


# In[8]:


TRAIN_PATH = f'{TGT_PATH}train/'
SUFFIXES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f')
if not os.path.exists(TRAIN_PATH):
    os.makedirs(TRAIN_PATH)
for s in SUFFIXES:
    p = f'{TRAIN_PATH}train_{s}/'
    if not os.path.exists(p):
        os.makedirs(p)


# In[9]:


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
        if label not in labels_set:
            return True
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
            df = df.append({'ImageID': f'train_{img[0]}/{img}', 'LabelName': label}, ignore_index=True)
        return df
    
    def something_to_submit(self):
        return len(self.images_labels) > 0
    
    def submit(self, executor):
        return executor.submit(do_job, self.img_id, self.sub_map)
        
def do_job(img_id, sub_map):
    processing_needed = False
    for bbox, tgt_img_id in sub_map.items():
        tgt_fname = f'{TRAIN_PATH}train_{tgt_img_id[0]}/{tgt_img_id}.jpg'
        if os.path.exists(tgt_fname):
            continue
        processing_needed = True
        break
    if not processing_needed:
        return
    
    fname = f'{RAW_PATH}train_{img_id[0]}/{img_id}.jpg'
    if not os.path.exists(fname):
        return
    img = PIL.Image.open(fname)
    w, h = img.size
    for bbox, tgt_img_id in sub_map.items():
        tgt_fname = f'{TRAIN_PATH}train_{tgt_img_id[0]}/{tgt_img_id}.jpg'
        if os.path.exists(tgt_fname):
            continue
        crop = (w * bbox[0], h * bbox[1], w * bbox[2], h * bbox[3])
        crop = list(map(int, crop))
        tgt_img = img.crop(crop)
        tgt_img.save(tgt_fname)


# In[46]:


NUM_JOBS = 6
MAX_CONCURRENT_JOBS = 10000
WAIT_SECONDS = 30
#df = df_bboxes[:30000]
df = df_bboxes
#ignore_groups = set(['0', '1'])
#ignore_groups = {'0', '1', '2', '3'} # 4 and 7 is fully unpacked, so, start it
ignore_groups = set()


# In[47]:


print("Converting %d bounding boxes" % len(df))
tgt_df = pd.DataFrame(columns=['ImageID', 'LabelName'])
fs = []
with futures.ThreadPoolExecutor(max_workers=NUM_JOBS) as executor:
    job = None
    try:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            img_id = row['ImageID']
            if img_id[0] in ignore_groups:
                continue
            if job is None:
                job = Job(img_id)
            if not job.add_row(row):
                if job.something_to_submit():
                    fs.append(job.submit(executor))
                    tgt_df = job.add_labels(tgt_df)
                job = Job(img_id)
                job.add_row(row)
                if len(fs) >= MAX_CONCURRENT_JOBS:                
                    done_fs, fs = futures.wait(fs, timeout=WAIT_SECONDS)
                    fs = list(fs)
                    print("Collected %d completed jobs, cur ImageID=%s" % (len(done_fs), img_id))
        if job.something_to_submit():
            fs.append(job.submit(executor))
            tgt_df = job.add_labels(tgt_df)
        tgt_df.to_csv(f'{TGT_PATH}train_proc.csv', index=False)
        print("Waiting for %d jobs to be completed" % len(fs))
        futures.wait(fs)
    except KeyboardInterrupt:
        print("Interrupt pressed, waiting for %d jobs to be completed gracefully" % len(fs))
        futures.wait(fs)

