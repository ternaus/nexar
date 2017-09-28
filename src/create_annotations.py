import pandas as pd
import os
from tqdm import tqdm

df = pd.read_csv('../data/train_boxes.csv')

df = df.rename(columns={'x0': 'xmin', 'y0': 'ymin', 'x1':'xmax', 'y1': 'ymax'})

# df['image_filename'] = df['image_filename'].str.replace('-1280_720', '').str.replace('-720_1280', '')

df['label'] = 'car'

g = df.groupby('image_filename')

os.mkdir('../data/annotations')

for file_name, df in tqdm(g):
    df.to_csv(os.path.join('../data', 'annotations', file_name.replace(r'.jpg', '.csv')), index=False)
