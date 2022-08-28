import random
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
import spacy
import sys, os
try:
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           os.path.pardir)))
except:
  sys.path.append(os.path.abspath(os.path.join("./",
                                           os.path.pardir)))

from lumi.stopwords  import stopwords
from lumi.gush_idx import *
from lumi.modeling_vlt5 import *
from lumi.tokenization_vlt5 import *
from lumi.modeling_dalle import *
from lumi.utils import *
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer, AutoModelWithLMHead
import torch
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import json
import tqdm
import numpy
import sys, os
import argparse


try:
  if minidalle is None:
    pass
except:
  minidalle = spacy_nlp = clip_model= clip_processor= stopwords_set= vlt5 = None
  device = 'cuda'

emotion_adj = ["surprised", "angry", "sad", "contemptous", "disgusted", "fearful", "happy"]
emotion_adj_set = set(emotion_adj)
shape_adj = ["near", "far", "large", "small", "medium", "tall", "broad", "crooked", \
             "curved", "deep", "even", "flat", "hilly", "jagged", \
              "round", "shallow", "square", "steep", "straight", "thick", \
              "thin", "triangular", "uneven"]
shape_adj_set = set(shape_adj)
color_adj = ["brown", "black", "blue", "gray", "green", \
             "pink", "purple", "red", "white", "yellow"] # "orange", confuses image generators to generate an orange fruit
color_adj_set = set(color_adj)
#TODO improve this with more variety

person_lst = ["man", "guy", "boy", "dude", "person", "woman", "lady", "gal", "girl",]
person_lst_set = set(person_lst)

def init_data(en_txt_gz_file, vlt5_data_file=None, pytorch_device = 'cuda'):
  global minidalle, spacy_nlp, clip_model, clip_processor, stopwords_set, vlt5, vlt5_data, device, vlt5_tokenizer, commongen_model, commongen_tokenizer
  device = pytorch_device
  if vlt5_data_file and vlt5_data is None:
      vlt5_data = torch.load(vlt5_data_file)
  if minidalle is None: 
    spacy_nlp = spacy.load('en_core_web_md')
  
    minidalle = DalleModel.from_pretrained("ontocord/minidalle").eval().half().to(device)
    vlt5 = VLT5.from_pretrained("ontocord/vlt5").eval().to(device) #half().
    #sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").half().eval().to(device)

    #labse =  SentenceTransformer("sentence-transformers/LaBSE").half().eval().to(device)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = clip_model.half().eval().to(device)
      
    vlt5_tokenizer = VLT5Tokenizer.from_pretrained("ontocord/vlt5")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    #vlt5_data = torch.load("/content/drive/Shareddrives/ontocord/vlt5_datasets/vlt5_data_0.pt")

    commongen_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-common_gen")
    commongen_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-common_gen").eval().half().to(device)

    stopwords_set = set(list(itertools.chain(*list(stopwords.values()))))

    if not os.path.exists("./en.txt.gz"):
      print (en_txt_gz_file)
      os.system(f"cp {en_txt_gz_file} ./en.txt.gz")
    IndexedGzipFileExt("en.txt.gz") 

def aug_obj(obj_str):
  obj =  " " +random.choice(["", "", "", "", "", "", "", "", ]+color_adj) + " " + obj_str
  obj =  obj.replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
  return obj.strip()

def aug_loc(loc_str=""):
  if loc_str and random.randint(0,3) != 0:
    loc =  random.choice(["", "", "", "", ]+shape_adj) + " " +random.choice(["", "", "", "", ]+color_adj) + " " + loc_str
  else:
    loc = loc_str +", the " +random.choice(["", "", "", "", ]+shape_adj) + " " + random.choice(["place", "location", "locale", "site",]) + " "
  loc =  loc.replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
  return loc.strip()

def aug_person(person_str="", is_male=True):
  norp = ""
  norp += " " +random.choice(["", "", "", "", "gay", "straight", "bisexual",])
  norp += " " +random.choice(["", "", "", "", "", "", "", "", ] + emotion_adj)
  norp += " " +random.choice(["", "", "", "", "conservative", "liberal", "moderate"])
  norp += " " +random.choice(["", "", "", "", "christian", "muslim", "buddhist", "hindu", ])
  norp += " " +random.choice(["", "", "", "",  "white", "black", "asian", "middle-eastern", "african", "hispanic", "native", "indian"])
  norp += " " +random.choice(["", "", "", "", "young", "middle-aged", "old"])
  if person_str and random.randint(0,1) == 0: 
    person = norp + " " + person_str
  else:
    if is_male: 
      person = "the " + norp + " " + random.choice(["man", "man", "man", "guy", "boy", "dude", "person"])
    else:
      person = "the " +  norp + " " + random.choice(["woman", "woman", "woman", "lady", "gal", "girl", "person"])
  person =  person.replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("old boy", "old person").replace("old girl", "old person").replace("middle-aged boy", "middle-aged person").replace("middle-aged girl", "middle-aged person")
  return person.strip()

def simplify_aug(sentence, all_aug):
  if type(all_aug) is dict:
    for key, val in all_aug.items():
      sentence = sentence.replace(key, val)
  return sentence


def augment_ents(l, do_person=True, do_loc=False, do_obj=False, simplify_person=True, prob_of_swap=.33, other_person_list=[]):
  global spacy_nlp
  aug2ent =  {}
  seen = {}
  doc = spacy_nlp(l)
  ents = [(e.text, e.label_) for e in doc.ents]
  if do_obj: 
    ents += [(e.text, 'OBJ') for e in doc.noun_chunks if len(e.text) > 4 and e.text.lower() not in stopwords_set] 
  for e_text, e_label in list(set(ents)):
    e_text = e_text.strip("()[]0123456789-:,.+? ")
    if not e_text: continue
    if e_text.lower() in seen: continue
    if random.random() > prob_of_swap: continue
    e_text = strip_left_stopwords(e_text)
    if not e_text.strip(): continue
    if e_text.lower() in seen: continue
    if e_label  in ('LOC', 'GPE', 'FAC',) and do_loc:
        aug_word =   aug_loc(e_text)
    elif e_label in ('PERSON',) and do_person:
        aug_word =  aug_person(e_text, random.randint(0,1))
    elif e_label in ('PRODUCT', 'EVENT', 'WORK_OF_ART', 'OBJ') and do_obj:
        aug_word =  aug_obj(e_text)
    else:
        aug_word = e_text
    l = l.replace(e_text, aug_word,1)
    aug2ent[aug_word] = e_text
    if e_label == 'PERSON' and simplify_person:
        aug_word_arr = aug_word.split()
        if len(aug_word_arr) > 3:
          aug_word2 = ("" if aug_word_arr[0] != "the" else "the") +" " + " ".join(aug_word_arr[-2:])
          aug2ent[aug_word] = aug_word2
    seen[e_text.lower()] = 1
  if other_person_list:
    for aug_word in other_person_list:
      aug_word_arr = aug_word.split()
      if len(aug_word_arr) > 3:
          aug_word2 = ("" if aug_word_arr[0] != "the" else "the") +" " + " ".join(aug_word_arr[-2:])
          aug2ent[aug_word] = aug_word2
  return l, aug2ent


def strip_left_stopwords(e_text):
  e_text2 = []
  add_rest = False
  for et in e_text.split():
      if add_rest or (et.lower() not in stopwords_set):
        add_rest = True
        e_text2.append(et)
  return " ".join(e_text2)

def get_decomposed_sent_to_img(matched_sentence, img, other_sent_arr=[]):
  global spacy_nlp, clip_model, clip_processor, minidalle, device, commongen_model, commongen_tokenizer
  doc = spacy_nlp(matched_sentence)
  noun_chunks = [strip_left_stopwords(e.text) for e in doc.noun_chunks if len(e.text) > 4 and e.text.lower() not in stopwords_set]
  ner_and_verbs = dict([(strip_left_stopwords(e.text.lower() if len(e.text) < 5 else e.text.lower()[:5]), e.text) for e in doc.ents if len(e.text) > 4] + \
                           [(strip_left_stopwords(e.text.lower() if len(e.text) < 5 else e.text.lower()[:5]), e.text) for e in doc if len(e.text) > 4 and e.tag_.startswith('VB') and e.text.lower() not in stopwords_set] + \
                           [(e.lower() if len(e) < 5 else e.lower()[:5], e) for e in noun_chunks ]) 
  text4 = list(set([a.strip("()[]0123456789-:,.+? ") for a in (list(ner_and_verbs.values()) + other_sent_arr) if a.strip()]))
  text4 = [a for a in text4 if a.strip()]
  if False: #to get ony longest subsuming text
    text5 = []
    text4.sort(key=lambda a: len(a), reversed=True)
    for atext in text4:
      if any(a for a in text5 if atext in a): continue 
      text5.append(atext)
    text4 = text5            
  if text4:
    #score the entities and verbs against the image
    clip_output = clip_image_to_multitext_score(clip_model, clip_processor, img, text4, decompose_image=True)  
    if clip_output is not None:
      #decomposed_image_features is shape=[1, 50, 512] dtype="float16"
      #image is shape = [100,100,3], dtype="uint8"
      #tokens is [1, 1028] int16
      most_similar_idx = clip_output['scores'].sort().indices[-1]
      sim1 = clip_output['scores'][most_similar_idx].item()
      matched_output = {'score': sim1, 'matched_sentence': matched_sentence, 'element2text': clip_output['element2text'], \
                              'decomposed_image_features': clip_output['decomposed_image_features'].cpu().numpy().tostring(), }
      return matched_output
  return None    

def create_qa_vlt5(matched_output, img, score_cutoff, aug2ent):
  global vlt5, vlt5_tokenizer
  ent2aug = dict([(b,a) for a, b in aug2ent.items()])
  element2text = matched_output['element2text']
  prev_element = ""
  if random.randint(0,3) == 0:
    answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is in the background?",  img)["text"]
    if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in ("nothing", "nowhere", "unknown", "black", "white")): 
      matched_output['qa'] = matched_output.get('qa',[]) +  [f"what is in the background?|| {answer}"]    
  if random.randint(0,3) == 0:
    answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is in the foreground?",  img)["text"]
    if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in ("nothing", "nowhere", "unknown", "black", "white")): 
      matched_output['qa'] = matched_output.get('qa',[]) +  [f"what is in the foreground?|| {answer}"]    
  if random.randint(0,3) == 0:
    answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is on the left?",  img)["text"]
    if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in ("nothing", "nowhere", "unknown", "black", "white")): 
      matched_output['qa'] = matched_output.get('qa',[]) +  [f"what is on the left?|| {answer}"]    
  if random.randint(0,3) == 0:
    answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is on the right?",  img)["text"]
    if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in ("nothing", "nowhere", "unknown", "black", "white")): 
      matched_output['qa'] = matched_output.get('qa',[]) +  [f"what is on the right?|| {answer}"]    
  if random.randint(0,3) == 0:
    answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is in this picture?",  img)["text"]
    if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in ("nothing", "nowhere", "unknown", "black", "white")): 
      matched_output['qa'] = matched_output.get('qa',[]) +  [f"what is in this picture?|| {answer}"]    
      
  for element, score in reversed(list(element2text.values())):
    if score >= score_cutoff and not element.endswith('ed'): 
      if prev_element:
        if random.randint(0,1) == 0:
          answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: where is {element} in relation to {prev_element}?", img)["text"]
          if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in ("nothing", "nowhere", "unknown", "black", "white")): 
            matched_output['qa'] = matched_output.get('qa',[]) +  [f"where is {element} in relation to {prev_element}?|| {answer}"]    
        else:
          answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: how is {element} and {prev_element} related?",  img)["text"]
          if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in ("nothing", "nowhere", "unknown", "black", "white")): 
            matched_output['qa'] = matched_output.get('qa',[]) +  [f"How is {element} and {prev_element} related?|| {answer}"]    
        
      if random.randint(0,1) == 0:
        answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is {element} doing?",  img)["text"]
        if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in ("nothing", "nowhere", "unknown", "black", "white")): 
            matched_output['qa'] = matched_output.get('qa',[]) +  [f"what is {element} doing?|| {answer}"]    
      elif random.randint(0,1) == 0:
        answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is {element} for?",  img)["text"]
        if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in ("nothing", "nowhere", "unknown", "black", "white")): 
            matched_output['qa'] = matched_output.get('qa',[]) +  [f"what is {element} for?|| {answer}"] 
      else:
        answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is {element}?",  img)["text"]
        if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in ("nothing", "nowhere", "unknown", "black", "white")): 
            matched_output['qa'] = matched_output.get('qa',[]) +  [f"what is {element}?|| {answer}"]    
      if any(a in color_adj_set for a in ent2aug.get(element, "").split()):
        answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what color is {element}?",  img) ["text"]
        if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in ("nothing", "nowhere", "unknown", "black", "white")): 
          matched_output['qa'] = matched_output.get('qa',[]) +  [f"what color is {element}?|| {answer}"]  
      if any(a in shape_adj_set for a in ent2aug.get(element, "").split()):
        answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what shape is {element}?",  img) ["text"]
        if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in ("nothing", "nowhere", "unknown", "black", "white")): 
          matched_output['qa'] = matched_output.get('qa',[]) +  [f"what shape is {element}?|| {answer}"]  
      if any(a in emotion_adj_set for a in ent2aug.get(element, "").split()):
        answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is {element} feeling?",  img) ["text"]
        if answer not in ("true", "false", "yes", "no") and (random.randint(0,5)==0 or answer not in ("nothing", "nowhere", "unknown", "black", "white")): 
          matched_output['qa'] = matched_output.get('qa',[]) +  [f"what is {element} feeling?|| {answer}"]  
      prev_element = element
    

def create_synthetic_text_image_data(output_append_to_file, input_en_txt_gz_file, max_items=10000, score_cutoff=0.20, max_img_per_doc=5, trimmed_text_word_len=50, verbose=False, pytorch_device='cuda'):
  global spacy_nlp, clip_model, clip_processor, minidalle, device, commongen_model, commongen_tokenizer
  init_data(input_en_txt_gz_file, pytorch_device=pytorch_device)
  with open(output_append_to_file, "a+") as out:
   with IndexedGzipFileExt("en.txt.gz") as f:
      for cnt in tqdm.tqdm(range(max_items)):
        j = random.randint(0, len(f)-1)
        l = f[j]
        l = l.decode().strip()
        if l == l.upper(): continue
        if "You are visiting the placeholder page" in l: continue
        l_lower = l.lower()
        if l_lower.count("free") + l_lower.count("dating") + l_lower.count("sex") + l_lower.count("fuck") + l_lower.count("cock") + l_lower.count("pussy") + l_lower.count("xxx") > 3: continue  
        
        if l_lower.count("free") +  l_lower.count("viagra") +   l_lower.count("cialis") > 3: continue 
        l = l.replace("because this is pdf file * PDF *", " ... ")
        l = l.replace("no short description", " ... ")
        l = l.replace("Field of the Invention", "") 
        l = l.replace("The present invention relates to", "")
        l = l.replace("Description of the Prior Art", "")
        l = l.replace("Prior Art", "")
        trimmed_text = l.lower().split()
        stopword_cnt = len([word for word in trimmed_text if word in stopwords_set ])
        shortword_cnt = len([word for word in trimmed_text if len(word) < 5])
        stopword_ratio = stopword_cnt/len(trimmed_text)
        shortword_ratio = shortword_cnt/len(trimmed_text)
        if not (stopword_ratio < 0.75 and random.random() < stopword_ratio and random.random() < shortword_ratio):
          continue
        qa = ""
        if "? ||" in l:
          l, answer = l.split("? ||",1)
          l_arr = l.split(".")
          l = ".".join(l_arr[:-1])
          qa = (l_arr[-1]+"?|| "+answer).strip()
        #TODO, detect public figures which we will replace with lower frequency
        other_person_list = []
        #gender swap to balance out the dataset
        if ("men " in l or " Men " in l or "men's " in l or " Men's " in l) and random.randint(0, 2) == 0:
          l = l.replace("men ", " women ").replace(" Men ", " Women ").replace("men's ", "women's ").replace(" Men's ", " Women's ")
          other_person_list.append("women")
        if ("man " in l or " Man " in l or "man's " in l or " Man's " in l) and random.randint(0, 2) == 0:
          l = l.replace("man ", "woman ").replace(" Man ", " Woman ").replace("man's ", "woman's ").replace(" Man's ", " Woman's ")
          other_person_list.append("woman")
        if (" boys " in l or " Boys " in l or " boys' " in l or " Boy's " in l) and random.randint(0, 2) == 0:
          l = l.replace(" boys ", " girls ").replace(" Boys ", " Girls ").replace(" boys' ", " girls' ").replace(" Boys' ", " Girls' ")
          other_person_list.extend(["girls"])
        if (" boy " in l or " Boy " in l or " boy's " in l or " Boy's " in l) and random.randint(0, 2) == 0:
          l = l.replace(" boy ", " girl ").replace(" Boy ", " Girl ").replace(" boy's ", " girl's ").replace(" Boy's ", " Girl's ")
          other_person_list.append("girl")
        if (" me " in l or " Me " in l) and random.randint(0, 2) == 0:
          l = l.replace(" me ", " her ").replace(" myself ", " herself ").replace(" I ", " she ").replace(" my ", " her ").replace(" mine ", " hers ").replace(" Me ", " Her ").replace(" My ", " Her ").replace(" Mine ", " Hers ").replace(" Myself ", " Herself ")
        elif (" you" in l or " You" in l) and random.randint(0, 2) == 0:
          l = l.replace(" you ", " she ").replace(" your ", " her ").replace(" yourself ", " herself ").replace(" yours ", " hers ").replace(" You ", " She ").replace(" Your ", " Her ").replace(" Yours ", " Hers ").replace(" Yourself ", " Herself ")
        elif (" he " in l or " He " in l) and random.randint(0, 2) == 0:
          l = l.replace(" he ", " she ").replace(" him ", " her ").replace(" himself ", " herself ").replace(" his ", " her ").replace(" He ", " She ").replace(" Him ", " Her ").replace(" His ", " Her ").replace(" Himself ", " Herself ")
        elif (" she " in l or " She " in l) and random.randint(0, 4) == 0:
          l = l.replace(" she ", " he ").replace(" her ", " him ").replace(" hers ", " his ").replace(" She ", " He ").replace(" Her ", " Him ").replace(" Hers ", " His ")
        l = l.replace("Huwoman", "Human").replace("huwoman", "human") #german, etc. needs to be fixed too.
        
        person = aug_person(person_str="", is_male=" he " in l or " He " in l)
        other_person_list.append(person)
        #augment the sentence with fake data
        l, aug2ent  = augment_ents(l, do_person=True, do_loc=False, do_obj=False, other_person_list=other_person_list)
        
        l = l.replace("  ", " ").replace("  ", " ").replace("  ", " ")
        l = l.replace("the the", "the").replace("The the", "The").replace("Dr. the", "the").replace("Mr. the", "the").replace("Mrs. the", "the").replace("Miss. the", "the").replace("Ms. the", "the")
        l = l.replace("Dr the", "the").replace("Mr the", "the").replace("Mrs the", "the").replace("Miss the", "the").replace("Ms the", "the")
        #print (l)
        #l = l.split()
        orig_l = l
        dat = []
        dat_cnt = 0
        trimmed_text_str_len= trimmed_text_word_len*6
        while True:
            dat_cnt += 1
            if dat_cnt >= max_img_per_doc*3: break
            if len(l) > trimmed_text_str_len:
              new_l = l[:trimmed_text_str_len].strip()
              if ". " in l[trimmed_text_str_len:] and l[trimmed_text_str_len:].index(". ") <= 15:
                idx = l[trimmed_text_str_len:].index(". ")
                new_l += l[trimmed_text_str_len:idx+trimmed_text_str_len] + "."
                l = l[trimmed_text_str_len+idx:]
              elif " " in l[trimmed_text_str_len:]:
                idx = l[trimmed_text_str_len:].index(" ")
                new_l += l[trimmed_text_str_len:idx+trimmed_text_str_len] 
                l = l[trimmed_text_str_len+idx:]
              else:
                l = l[trimmed_text_str_len:]
              new_l  = new_l.strip(".").replace("..", ".").replace("..", ".").replace("..", ".").replace("  ", " ")
              if person and random.randint(0,3) == 0:
                new_l = new_l.replace(" he ", " "+person+" ",1).replace(" He ", " "+person+" ",1).replace(" him ", " "+person+" ",1).replace(" Him ", " "+person+" ",1).replace(" she ", " "+person+" ",1).replace(" She ", " "+person+" ",1).replace(" her ", " "+person+" ",1).replace(" Her ", " "+person+" ",1)
              if person and random.randint(0,3) == 0:
                new_l = new_l.replace(" his ", " "+person+"'s ",1).replace(" His ", " "+person+"'s ",1).replace(" hers ", " "+person+"'s ",1).replace(" Hers ", " "+person+"'s ",1)
              if len(new_l.strip())>  20:
                dat.append(new_l)
            else:
              new_l = l.strip()
              new_l  = new_l.strip(".").replace("..", ".").replace("..", ".").replace("..", ".").replace("  ", " ")
              if person and random.randint(0,2) == 0:
                new_l = new_l.replace(" he ", " "+person+" ",1).replace(" He ", " "+person+" ",1).replace(" him ", " "+person+" ",1).replace(" Him ", " "+person+" ",1).replace(" she ", " "+person+" ",1).replace(" She ", " "+person+" ",1).replace(" her ", " "+person+" ",1).replace(" Her ", " "+person+" ",1)
              if person and random.randint(0,2) == 0:
                new_l = new_l.replace(" his ", " "+person+"'s ",1).replace(" His ", " "+person+"'s ",1).replace(" hers ", " "+person+"'s ",1).replace(" Hers ", " "+person+"'s ",1)
              if len(new_l.strip())>  20:
                dat.append(new_l)
              break

        dat_cnt = 0
        for didx, text in enumerate(dat):
            if not text.strip(): continue
            if dat_cnt >= max_img_per_doc: break
            trimmed_text = text.split()
            with torch.no_grad():
              trimmed_text = " ".join(trimmed_text)
              tokens, img = minidalle.generate(trimmed_text, image_output=True, token_output=True)
              img = img.resize((100,100))
              tokens = tokens.cpu().numpy()
              tokens.dtype = np.int16
              vlt5_output = vlt5_image2text(vlt5, vlt5_tokenizer, "caption:",  img, annotated_image=True,
                                    min_length=max(4, len(trimmed_text.split())-10),  
                                    max_length=4+len(trimmed_text.split())*10, 
                                    no_repeat_ngram_size=2)
              vlt5_caption = vlt5_output['text'].strip(".").replace("..", ".").replace("..", ".")
              if "." in vlt5_caption:
                vlt5_caption, _ = vlt5_caption.split(".",1)
              vlt5_caption = " ".join([e for e in vlt5_caption.split() if len(e) <= 10])
              vlt5_caption = vlt5_caption.replace("painting of ", "").replace("photograph of ", "").replace("photo of ", "").replace("picture of ", "").replace("photo of ", "").replace("drawing of ", "").replace("illusration of ", "")
              vlt5_caption = vlt5_caption.replace("of the painting", "").replace("of the photograph", "").replace("of the photo", "").replace("of the picture", "").replace("of the photo", "").replace("of the drawing", "").replace("of the illusration", "")
              if "clock" in vlt5_caption and random.randint(0,5) != 0: continue #minidalle sometimes creates pictures of watches when it can't figure out what to draw
              #print (vlt5_caption, '**', trimmed_text)
              text2 = trimmed_text
              text2_arr = text2.replace("?", ". ").replace("!", ". ").replace("- ", ". ").replace(";", ". ").strip(".").replace("..", ".").replace("..", ".").replace("..", ".").replace("  ", " ").split(". ")
              text3 = [t.strip() for t in  text2_arr]
              if len(text2_arr) > 1: text3 = text3 + [text2]
              #text3 = [vlt5_caption] + text3
              # find the sentence that is most like this picture, the last item being the complete text chunk
              sim1 = 0.0
              clip_output = clip_image_to_multitext_score(clip_model, clip_processor, img, text3)
              if clip_output is None: continue
              most_similar_idx = clip_output['scores'].sort().indices[-1]
              sim1 = clip_output['scores'][most_similar_idx].item()
              # clip scores of generated images tend to be lower; this filters out really bad matches
              if sim1 > score_cutoff:    
                matched_sentence = text3[most_similar_idx]
                if not matched_sentence.strip(): continue
                prev_text = next_text = ""
                if most_similar_idx < len(text3)-1:
                  if matched_sentence not in text3[-1]:
                    print ("problem",matched_sentence,'**', text3[-1])
                  else:
                    prev_text, next_text = text3[-1].split(matched_sentence,1)
                    if matched_sentence in prev_text or matched_sentence in next_text:
                      print ("doubled", matched_sentence)
                prev_text2 = "" if didx <= 0  else dat[didx-1]
                if prev_text and prev_text2 and prev_text[0] == prev_text[0].upper():
                  prev_text = simplify_aug((prev_text2+". "+prev_text).strip(" ."), aug2ent)
                else:
                  prev_text = simplify_aug((prev_text2+" "+prev_text).strip(" ."), aug2ent)
                  next_text2 = "" if didx >= len(dat) -1 else  dat[didx+1]
                  if next_text2 and next_text and next_text2[0] == next_text2[0].upper():
                    next_text = simplify_aug((next_text +". "+next_text2).strip(" ."), aug2ent)
                  else:
                    next_text = simplify_aug((next_text +" "+ next_text2).strip(" ."), aug2ent)  
                
                #let's do some cleanup of the ents since we injected more information then is in natural text
                matched_sentence = simplify_aug(matched_sentence, aug2ent)
                # now find the entities and important verbs in the most similar sentence
                matched_output = get_decomposed_sent_to_img(matched_sentence, img, [vlt5_caption])
                if matched_output:
                    matched_output['tokens'] = tokens.tostring()
                    matched_output['thumbnail'] = np.array(img).tostring()
                    matched_output['prev_text'] = prev_text
                    matched_output['next_text'] = next_text
                    if qa:
                      matched_output['qa'] = matched_output.get('qa',[]) + qa
                    element2text = matched_output['element2text']
                    vlt5_caption_with_score = [e for e in element2text.values() if e[0] == vlt5_caption]
                    if vlt5_caption_with_score:
                      if vlt5_caption_with_score[0][1] > score_cutoff*1.25:
                        matched_output['qa'] = matched_output.get('qa',[]) +  [f"Is there {vlt5_caption_with_score[0][0]} in this picture? || Yes"]
                        #create question-answers from image
                        create_qa_vlt5(matched_output, img, score_cutoff*1.25,  aug2ent)
                      elif random.randint(0,5)==0:
                        matched_output['qa'] = matched_output.get('qa',[]) +  [f"Is there {vlt5_caption_with_score[0][0]} in this picture? || No"]  
                         
                    if verbose:
                      print ( matched_output['score'], '**', matched_output['matched_sentence'], '***', aug2ent, '***', matched_output['element2text'], '***', matched_output.get('qa'))
                      if 'annotated_image' in vlt5_output['frcnn_output']:
                        display(vlt5_output['frcnn_output']['annotated_image'])
                      else:
                        display(img)
                    out.write(str(matched_output)+"\n")
                    dat_cnt += 1    
                    
                    #now let's create a different sentence based on the elements of the previous sentence
                    word_str = ", ".join([a[0] for a in element2text.values() if a[1] >= score_cutoff and a[0] != vlt5_caption])
                    if word_str:
                      
                      generated_sentence = commongen_model.generate(commongen_tokenizer.encode(word_str, return_tensors="pt").to(device), 
                                                                    min_length=len(word_str.split())*3, 
                                                                    max_length=len(word_str.split())*10, 
                                                                    no_repeat_ngram_size=2, )
                      generated_sentence = commongen_tokenizer.decode(generated_sentence[0], skip_special_tokens=True).strip(". ")
                      if ".." in generated_sentence: generated_sentence, _ = generated_sentence.split("..", 1)
                      generated_sentence = generated_sentence.strip()
                      l_lower = generated_sentence.lower()
                      if l_lower.count(" sex ") + l_lower.count(" fuck ") + l_lower.count(" cock ") + l_lower.count(" pussy ") + l_lower.count(" xxx ") > 1: continue  
                      if "," in generated_sentence and generated_sentence.count(",") > len(generated_sentence.split())*.5: continue
                      orig_generated_sentence = generated_sentence
                      
                      #augment the sentence with fake data
                      image_type = random.choice(["", "",  "", "", "",  "", "rendering of :", "vector art of: ", "scene of: ", "movie still of: ", "textbook illustration of: ", "realistic drawing: ", "picture of: ", "sketch of: ", "cartoon of: ", "painting of: "])
                      if not ("rendering" in image_type "art" in image_type or "cartoon" in image_type or "illustration" in image_type or "drawing" in image_type or "sketch" in image_type):
                        mult = 1.0
                        generated_sentence, aug2ent  = augment_ents(generated_sentence, do_person=False, do_loc=True, do_obj=True, other_person_list=other_person_list)
                      else:
                        multi = 1.25
                      generated_sentence = image_type +generated_sentence
                      #generate an image 
                      tokens, img = minidalle.generate(generated_sentence, image_output=True, token_output=True)
                      img = img.resize((100,100))
                      tokens = tokens.cpu().numpy()
                      tokens.dtype = np.int16
                      print ("generated augmented", generated_sentence)
                      generated_sentence = orig_generated_sentence
                      matched_output2 = get_decomposed_sent_to_img(generated_sentence, img)
                      if matched_output2 and matched_output2['score'] >= multi*score_cutoff and len(a for a in matched_output2['element2text'].values() if a[1] >= score_cutoff) >= (len(matched_output2['element2text'])*.5):
                        matched_output['tokens2'] = tokens.tostring()
                        matched_output['thumbnail2'] = np.array(img).tostring()
                        matched_output['element2text2'] = matched_output2['element2text']
                        matched_output['score2'] = matched_output2['score']
                        matched_output['decomposed_image_features2'] = matched_output2['decomposed_image_features2']
                        matched_output['matched_sentence2'] = matched_output2['matched_sentence']
                        if verbose:
                            print ( matched_output2['score'], '**', matched_output2['matched_sentence'],  '***', aug2ent, '***', matched_output['element2text'])
                            display(img)  
                        dat_cnt += 1
                        out.write(str(matched_output)+"\n")
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Text Synthetic Data')
    
    parser.add_argument('-output_append_to_file', dest='output_append_to_file', type=str, help='File to write image/text data to')
    parser.add_argument('-input_en_txt_gz_file', dest='input_en_txt_gz_file', type=str, help='Input english text file, in .gz format')
    parser.add_argument('-max_items', dest='max_items', type=int, help='Maximum items to create', default=10000)
    parser.add_argument('-max_img_per_doc', dest='max_img_per_doc', type=int, help='Maximum images to create from one document', default=5)
    parser.add_argument('-score_cutoff', dest='score_cutoff', type=float, help='Cutoff score for image/text matching using CLIP. Usually around .23-.20', default=.2)
    parser.add_argument('-trimmed_text_word_len', dest='trimmed_text_word_len', type=int, help='The approximate number of words per sentence used to generate images', default=50)
    parser.add_argument('-pytorch_device', dest='pytorch_device', type=str, help='the device', default= "cuda")
    
    args = parser.parse_args()
    create_synthetic_text_image_data(output_append_to_file=args.output_append_to_file, \
                                     input_en_txt_gz_file=args.input_en_txt_gz_file, \
                                     max_items=args.max_items, \
                                     score_cutoff=args.score_cutoff, \
                                     max_img_per_doc=args.max_img_per_doc, \
                                     trimmed_text_word_len=args.trimmed_text_word_len, \
                                     verbose=False, \
                                     pytorch_device=args.pytorch_device)
 
