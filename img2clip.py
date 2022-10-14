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

def save_clip_batch(filename, imgs, idxs, mmap_len, cls_weight=.9,):
  with torch.no_grad():
    p = next(clip_model.parameters())
    inputs = clip_processor(images=imgs, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].to(dtype=p.dtype, device=p.device)
    inputs['return_dict'] = True
    clip_vision_output = clip_model.vision_model(**inputs)
    o = (clip_vision_output["last_hidden_state"][0,1:,:] + cls_weight*10*clip_vision_output["last_hidden_state"][0,0,:])/(cls_weight*10+1)
    print ('A', o.shape)
    decomposed_image_features = clip_model.visual_projection(clip_model.vision_model.post_layernorm(o))
    # image_features[0] is the main picture, the rest are the box parts of the picture
    print ('B', decomposed_image_features.shape)
    image_features = clip_model.visual_projection(clip_vision_output["pooler_output"]) 
    print (image_features.shape, decomposed_image_features.shape)
    return image_features, decomposed_image_features 

if __name__ == "__main__":

  md5_hash ={}
  dat = pd.read_csv('./laion_subset_dedup.tsv.gz', sep='\t', header=None)
  text = dat[0].tolist()
  kw = dat[1].tolist()
  url = [url if url.startswith("http") and "," not in url else "" for url in dat[2].tolist()]
  url2dat = dict([(str(u).strip(), [str(t).strip(), str(k).strip(), None]) for u, t, k in zip(url, text, kw)])
  all_files = []
  mmap_len = 1
  seen_files = {}
  batch = []
  with open("laion_subset_dedup2.tsv", "w", encoding="utf8") as out:
    while True:
      all_files = []
      for d in glob.glob("./images/*"):
        if os.path.isdir(d):
          for f in glob.glob(d+"/*.json"):
            all_files.append(f)
            if f in seen_files: continue
            seen_files[f] = 1
      for f in tqdm.tqdm(all_files):
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
            dat2[2] = 1
            img = Image.open (f.replace(".json", ".jpg"))
            img_data =  np.array(img).flatten()
            np_memmap("./laion_subset_dedup2_images.mmap", shape=[mmap_len, len(img_data)], idxs=[mmap_len-1], dat=img_data, dtype=img_data.dtype)
            mmap_len += 1
            out.write(dat2[0]+"\t"+dat2[1]+"\n")
            if len(batch) > 10:
              ret = save_clip_batch(None, batch, None, None, cls_weight=.9,)
              batch = []
            batch.append(img)
      time.sleep(60)
      if batch:
        ret = save_clip_batch(None, batch, None, None, cls_weight=.9,)
              
