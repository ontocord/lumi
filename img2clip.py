import random
import spacy
import sys, os
try:
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           os.path.pardir)))
except:
  sys.path.append(os.path.abspath(os.path.join("./",
                                           os.path.pardir)))

from riverbed.stopwords  import stopwords
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer, AutoModelWithLMHead
import torch
from torch.nn.functional import cosine_similarity
import json
import tqdm
import numpy
import sys, os
import argparse
import glob
import torch
import pandas as pd
import numpy as np
import time
from img2dataset import download
from multiprocessing import Process

device = 'cuda'
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model = clip_model.half().eval().to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
def np_memmap(filename, dat=None, idxs=None, shape=None, dtype=np.float16, offset=0, order='C' ):
  if not filename.endswith(".mmap"):
    filename = filename+".mmap"
  if os.path.exists(filename):
    mode = "r+"
  else:
    mode = "w+"
  if shape is None and dat is not None: 
    shape = dat.shape
  if not shape: shape = [0, 1]
  memmap = np.memmap(filename, mode=mode, dtype=dtype, shape=tuple(shape), offset=offset, order=order)
  if dat is None:
    return memmap
  if tuple(shape) == tuple(dat.shape):
    memmap[:] = dat
  else:
    memmap[idxs] = dat
  return memmap

def save_clip_batch(decomposed_image_features_mmap, image_features_mmap, imgs, idxs, mmap_len, cls_weight=.9,):
  with torch.no_grad():
    p = next(clip_model.parameters())
    inputs = clip_processor(images=imgs, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].to(dtype=p.dtype, device=p.device)
    inputs['return_dict'] = True
    clip_vision_output = clip_model.vision_model(**inputs)
    o = (clip_vision_output["last_hidden_state"][:,1:,:] + cls_weight*10*clip_vision_output["last_hidden_state"][:,0,:].unsqueeze(1)/(cls_weight*10+1))    
    decomposed_image_features = clip_model.visual_projection(clip_model.vision_model.post_layernorm(o))
    shape = list(decomposed_image_features.shape)
    shape[0] = mmap_len
    np_memmap(decomposed_image_features_mmap, dat=decomposed_image_features.cpu().numpy(), idxs=idxs, shape=shape)
    image_features = clip_model.visual_projection(clip_vision_output["pooler_output"]) 
    shape = list(image_features.shape)
    shape[0] = mmap_len
    np_memmap(image_features_mmap, dat=image_features.cpu().numpy(), idxs=idxs, shape=shape)
    return image_features, decomposed_image_features 

def img2clip_download(url_list, image_size):
  download(url_list="./url.txt", image_size=image_size)

def create_img2clip_data_mmaps(laion_df=None, image_size=100, shard_range=None):
  md5_hash ={}
  if laion_df is None:
    laion_df = pd.read_csv('./laion_subset_dedup.tsv.gz', sep='\t', header=None)
  text = laion_df[0].tolist()
  kw = laion_df[1].tolist()
  url = [url if url.startswith("http") and "," not in url else "" for url in laion_df[2].tolist()]
  if shard_range is None:
    shard_range = [0, len(shard_range)]
  shard_name = str(shard_range[0])+"_"+str(shard_range[1])
  open("./url.txt", "w").write("\n".join(url[shard_range[0]:shard_range[1]+1]))
  url2dat = dict([(str(u).strip(), [str(t).strip(), str(k).strip(), None]) for u, t, k in zip(url, text, kw)])
  p = Process(target=img2clip_download, args=("./url.txt", image_size))
  p.start()
  time.sleep(10)
  all_files = []
  mmap_len = 1
  seen_files = {}
  batch = []
  idxs = []
  with open(f"./laion_subset_dedup_{shard_name}.tsv", "w", encoding="utf8") as out:
    while True:
      all_files = []
      for d in glob.glob("/content/images/*"):
        if os.path.isdir(d):
          for f in glob.glob(d+"/*.json"):
            if f in seen_files: continue
            seen_files[f] = 1
            all_files.append(f)
      if not all_files: 
        if p.is_alive(): 
          time.sleep(10)
          continue
        else:
          break
      for f in all_files:
        file_dat = json.load(open(f))
        if file_dat["status"] == "success":
          url = file_dat["url"]
          if url in url2dat and url2dat[url][2] is not None: continue
          md5_str = file_dat["md5"]
          if md5_str in md5_hash and url: 
            if url in url2dat: del url2dat[url]
            continue
          md5_hash[md5_str] = 1
          dat2 = url2dat.get(url)
          if not dat2:
            print ('problem', url)
          else: 
            try:
              img = Image.open (f.replace(".json", ".jpg"))
            except:
              continue
            dat2[2] = 1
            img_data =  np.array(img).flatten()
            np_memmap(f"./laion_subset_dedup2_images_{shard_name}.mmap", shape=[mmap_len, len(img_data)], idxs=[mmap_len-1], dat=img_data, dtype=img_data.dtype)
            out.write(dat2[0]+"\t"+"" if dat2[1] == "nan" else dat2[1]+"\n")
            if len(batch) > 2000:
              ret = save_clip_batch(f"./laion_clip_{shard_name}.mmap", f"./laion_decomposed_clip_{shard_name}.mmap", imgs=batch, idxs=idxs, mmap_len=mmap_len, cls_weight=.9,)
              batch = []
              idxs=[]
            batch.append(img)
            idxs.append(mmap_len-1)
            mmap_len += 1
        if batch:
          ret = save_clip_batch(f"./laion_clip_{shard_name}.mmap", f"./laion_decomposed_clip_{shard_name}.mmap", imgs=batch, idxs=idxs, mmap_len=mmap_len, cls_weight=.9,)
      time.sleep(10)
  p.join()

