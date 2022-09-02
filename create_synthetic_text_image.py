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
shape_adj = ["banana-shaped", "strawberry-shaped", "grapes-shaped", "apple-shaped", "watermelon-shaped", "orange-shaped", "blueberry-shaped", 
             "lemon-shaped", "large", "small", "medium", "tall", "broad", "crooked", \
             "curved", "deep", "even", "flat", "hilly", "jagged", \
              "round", "shallow", "square", "steep", "straight", "thick", \
              "thin", "triangular", "uneven"]
shape_adj_set = set(shape_adj)
color_adj = ["brown", "black", "blue", "gray", "green", "pink", "purple", "red", "white", "yellow",] #orange confuses image generators to generate an orange fruit
color_adj_set = set(color_adj)
#TODO improve this with more variety

person_lst = ["man", "guy", "boy", "dude", "person", "woman", "lady", "gal", "girl",]
person_lst_set = set(person_lst)
age_adj_lst = ["young", "teen", "young-adult", "middle-aged", "old"]
age_adj_set = set(age_adj_lst)

religion_lst = ["christian", "muslim", "buddhist", "hindu"]
religion_lst_set = set(religion_lst)
race_lst = ["white", "black", "asian", "middle-eastern", "african", "hispanic", "native", "indian"]
race_lst_set = set(race_lst)
sexual_orientation_lst = ["gay", "straight", "bisexual",]
sexual_orientation_lst_set = set(sexual_orientation_lst)
political_affiliation_lst = ["conservative", "liberal", "moderate"]
political_affiliation_lst_set = set(political_affiliation_lst)

common_vlt5_words = ("background", "foreground", "left", "right", "nothing", "nowhere", "unknown", "black", "white")

mood_lst = ["cheerful", "reflective", "gloomy", "humorous", "melancholy", "idyllic", \
                      "whimsical", "romantic", "mysterious", "ominous", "calm", "lighthearted", \
                      "hopeful", "angry", "fearful", "tense", "lonely"]
image_type_lst = ["rendering", "vector-art ", "scene", "movie-still", \
                      "textbook-illustration", "realistic-drawing", "sketch", "cartoon", "painting"]
                      
    
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
  norp += " " +random.choice(["", "", "", "", "", "", "", "", ] + sexual_orientation_lst + political_affiliation_lst  + emotion_adj)
  norp += " " +random.choice(["", "", "", "", "", "", "", "", ] + religion_lst + race_lst)
  norp += " " +random.choice(["", "", "", "", ] + age_adj_lst)
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
    lst = list(all_aug.items())
    lst.sort(key=lambda a: len(a[0]), reverse=True)
    for key, val in lst:
      sentence = sentence.replace(key, val)
  return sentence

def re_augment(sentence, all_aug):
  if type(all_aug) is dict:
    lst = list(all_aug.items())
    lst.sort(key=lambda a: len(a[0]), reverse=True)
    for i, key_val in enumerate(lst):
      sentence = sentence.replace(key_val[1], '**'+str(i)+'**')
    for i, key_val in enumerate(lst):
      sentence = sentence.replace('**'+str(i)+'**', key_val[0])      
  return sentence

def augment_ents(l, do_person=True, do_loc=False, do_obj=False, simplify_person=True, prob_of_swap=.33, other_person_list=[]):
  global spacy_nlp
  def get_person_questions(aug_word, qa_list):
      the_person =  " ".join(aug_word.split()[-2:])
      if the_person not in person_lst_set: the_person = "person"
      emotion = [a for a in aug_word.split() if a in emotion_adj_set]
      if emotion: qa_list.append((the_person, f"what is {the_person} feeling?||{emotion[0]}"))
      age = [a for a in aug_word.split() if a in age_adj_set]
      if age: qa_list.append((the_person, f"how old is {the_person}?||{age[0]}"))
      the_person = the_person.split()[-1]
      religion = [a for a in aug_word.split() if a in religion_lst_set]
      if religion: qa_list.append((the_person, f"what religion is {the_person}?||{religion[0]}"))
      race = [a for a in aug_word.split() if a in race_lst_set]
      if race: qa_list.append((the_person, f"what race is {the_person}?||{race[0]}"))
      if "person" not in the_person:
        qa_list.append((the_person, f"what gender is the person?||{the_person}"))
        
  qa_list = []
  aug2ent =  {}
  seen = {}
  doc = spacy_nlp(l)
  ents = [(e.text, e.label_) for e in doc.ents]
  #TODO, use nltk.wordnet to see if the parent node is an obj vs. abstract.
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
        thing = " ".join(aug_word.split()[-2:])
        color = [a for a in aug_word.split() if a in color_adj_set]
        if color: qa_list.append((thing, f"what color is {thing}?||{color[0]}"))
        shape = [a for a in aug_word.split() if a in shape_adj_set]
        if shape: qa_list.append((thing, f"what shape is {thing}?||{shape[0]}"))
    elif e_label in ('PERSON',) and do_person:
        aug_word =  aug_person(e_text, random.randint(0,1))
        get_person_questions(aug_word, qa_list)        
    elif e_label in ('PRODUCT', 'EVENT', 'WORK_OF_ART', 'OBJ') and do_obj:
        aug_word =  aug_obj(e_text)
        thing =  " ".join(aug_word.split()[1:])
        color = [a for a in aug_word.split() if a in color_adj_set]
        if color: qa_list.append((thing, f"what color is {thing}?||{color[0]}"))
    else:
        aug_word = e_text
    l = l.replace(e_text, aug_word,1)
    aug2ent[aug_word] = e_text
    if e_label == 'PERSON' and simplify_person:
        aug_word_arr = aug_word.split()
        if e_text.split()[-1] == aug_word_arr[-1]:
          aug_word_arr[-1] = "person"
        if len(aug_word_arr) > 3:
          aug_word2 = ("" if aug_word_arr[0] != "the" else "the") +" " + " ".join(aug_word_arr[-2:])
        else:
          aug_word2 = " ".join(aug_word_arr)
        aug2ent[aug_word] = aug_word2
    seen[e_text.lower()] = 1
    
  if other_person_list:
    for aug_word in other_person_list:
      get_person_questions(aug_word, qa_list)
      aug2ent[aug_word] = aug_word
      aug_word_arr = aug_word.split()
      if len(aug_word_arr) > 3 and simplify_person:
          aug_word2 = ("" if aug_word_arr[0] != "the" else "the") +" " + " ".join(aug_word_arr[-2:])
          aug2ent[aug_word] = aug_word2
  
  l = l.replace("  ", " ").replace("  ", " ").replace("  ", " ")
  l = l.replace(" an the", " the").replace(" a the", " the").replace("the the", "the").replace("The the", "The").replace("Dr. the", "the").replace("Mr. the", "the").replace("Mrs. the", "the").replace("Miss. the", "the").replace("Ms. the", "the")
  l = l.replace("Dr the", "the").replace("Mr the", "the").replace("Mrs the", "the").replace("Miss the", "the").replace("Ms the", "the")          
  return l, aug2ent, qa_list


def strip_left_stopwords(e_text):
  e_text2 = []
  add_rest = False
  for et in e_text.split():
      if add_rest or (et.lower() not in stopwords_set):
        add_rest = True
        e_text2.append(et)
  return " ".join(e_text2)

def get_sent_to_img(matched_sentence, img, other_sent_arr=[], get_cropped_images=False, num_boxes=5):
  global spacy_nlp, clip_model, clip_processor, minidalle, device, commongen_model, commongen_tokenizer
  doc = spacy_nlp(matched_sentence)
  noun_chunks = [strip_left_stopwords(e.text) for e in doc.noun_chunks if len(e.text) > 4 and e.text.lower() not in stopwords_set]
  verbs = [(strip_left_stopwords(e.text.lower() if len(e.text) < 5 else e.text.lower()[:5]), e.text) for e in doc if len(e.text) > 4 and e.tag_.startswith('VB') and e.text.lower() not in stopwords_set]                            
  ner_and_verbs = dict([(strip_left_stopwords(e.text.lower() if len(e.text) < 5 else e.text.lower()[:5]), e.text) for e in doc.ents if len(e.text) > 4] + \
                           verbs + \
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
    if get_cropped_images:
      normalized_boxes = decode_image(asarray(img), vlt5.frcnn,  vlt5.image_preprocessor, max_detections=num_boxes)["normalized_boxes"][0]
      #score the entities and verbs against the image
      clip_output = clip_image_to_multitext_score(clip_model, clip_processor, img, text4, decompose_image=True, normalized_boxes=normalized_boxes, ignore_from_crop=verbs+other_sent_arr)  
    else:
      clip_output = clip_image_to_multitext_score(clip_model, clip_processor, img, text4, decompose_image=True, ignore_from_crop=verbs+other_sent_arr)  

    if clip_output is not None:
      #text2image_scores = dict([(text4[idx], clip_output['scores'][idx].item()) for idx in range(len(text4))]) 
      #most_similar_idx = clip_output['scores'].sort().indices[-1]
      #sim1 = clip_output['scores'][most_similar_idx].item()
      matched_output = {'matched_sentence': matched_sentence, 'cropped2text': clip_output['cropped2text'], \
                              'cropped_image_features': None if clip_output['cropped_image_features'] is None else clip_output['cropped_image_features'].cpu().numpy().tostring(),  \
                              'decomposed2text': clip_output['decomposed2text'], \
                              'decomposed_image_features': None if clip_output['decomposed_image_features'] is None else  clip_output['decomposed_image_features'].cpu().numpy().tostring(),\
                              'image_features': clip_output['image_features'].cpu().numpy().tostring(),\
                       }
      return matched_output, clip_output['cropped_images']
  return None, None    

def create_qa_from_vlt5(l, img,  aug2ent, max_qa=10, potential_qa_list=None):
    if potential_qa_list is None: potential_qa_list = []
    prev_element = "" 
    if " woman " in l:
      person = "woman"
    elif " girl " in l:
      person = "girl"
    elif " boy " in l:
      person = "boy"
    elif " man " in l:
      person = "man"
    elif " person " in l:
      person = "person"
    else:
      person = ""
    if person:
        answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is the {person} feeling?",  img)["text"]
        if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in common_vlt5_words): 
          potential_qa_list.append((person, f"what is {person} feeling?||{answer}"))
    entity_to_qa = 0    
    elements = list(aug2ent.values())
    elements.sort(key=lambda a: len(a), reverse=True)
    description = ""
    answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is in this picture?",  img)["text"]
    if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in common_vlt5_words): 
      description = answer
      potential_qa_list.append((answer, f"what is in this picture?||{answer}"))
      elements = [description] + elements
    answer2qa = {}
    for element in elements:
        if element != description and element not in l: continue
        if entity_to_qa >= max_qa: break
        color = [a for a in element.split() if a in color_adj_set]
        shape = [a for a in element.split() if a in shape_adj_set]
        if element == description:
          answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: where is {element}?",  img)["text"]
          if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in common_vlt5_words): 
              potential_qa_list.append((element, f"where is {element}?||{answer}"))
              entity_to_qa +=1
          answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is {element} doing?",  img)["text"]
          if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in common_vlt5_words): 
              potential_qa_list.append((element, f"what is {element} doing?||{answer}"))
              entity_to_qa +=1
              if answer.endswith("ing"):
                act = answer
                prep = random.choice(['with','from','to','at','in'])
                answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is {element} {act} {prep}?",  img)["text"]
                if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in common_vlt5_words): 
                    potential_qa_list.append((element + ' and ' + act, f"what is {element} {act} {prep}?||{answer}"))
                    entity_to_qa +=1
        elif shape and random.randint(0,3) == 0: 
          answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what shape is {element}?",  img)["text"]
          if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in ("nothing", "nowhere", "unknown", "black", "white")): 
              potential_qa_list.append((element, f"what shape is {element}?||{answer}"))
              entity_to_qa +=1
        elif color and random.randint(0,3) == 0: 
          answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what color is {element}?",  img)["text"]
          if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in common_vlt5_words): 
              potential_qa_list.append((element, f"what color is {element}?||{answer}"))
              entity_to_qa +=1
        elif random.randint(0,1) == 0 and not (element.endswith("ed") or element.endswith("ing") or element.endswith("s")):
          answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is {element} doing?",  img)["text"]
          if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in common_vlt5_words): 
              potential_qa_list.append((element, f"what is {element} doing?||{answer}"))
              entity_to_qa +=1
              if answer.endswith("ing"):
                act = answer
                prep = random.choice(['with','from','to','at','in'])
                answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is {element} {act} {prep}?",  img)["text"]
                if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in common_vlt5_words): 
                    potential_qa_list.append((element + ' and ' + act, f"what is {element} {act} {prep}?||{answer}"))
                    entity_to_qa +=1
        elif random.randint(0,1) == 0:
          if  element.endswith("ing"):
            answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is {element}?",  img)["text"]
            if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in common_vlt5_words): 
                potential_qa_list.append((element, f"what is {element}?||{answer}"))
                entity_to_qa +=1
          else:
            answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what is {element} for?",  img)["text"]
            if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in common_vlt5_words): 
                potential_qa_list.append((element, f"what is {element} for?||{answer}"))
                entity_to_qa +=1
        elif random.randint(0,1) == 0 and prev_element:
          answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: where is {element} and {prev_element}?",  img)["text"]
          if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in common_vlt5_words): 
              potential_qa_list.append((element+' and '+ prev_element, f"where is {element} and {prev_element}?||{answer}"))
              entity_to_qa +=1
        elif random.randint(0,1) == 0:
          answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: where is {element}?",  img)["text"]
          if answer not in ("true", "false", "yes", "no") and (random.randint(0,2)==0 or answer not in common_vlt5_words): 
              potential_qa_list.append((element, f"where is {element}?||{answer}"))
              entity_to_qa +=1            
        prev_element = element
    return list(set(potential_qa_list))
                    
def create_qa(matched_output, img, score_cutoff, potential_qa_list=[], high_score_mult=1.2):
  global vlt5, vlt5_tokenizer
  l = matched_output['matched_sentence']
  ent2score = {}
  if True:
    decomposed2text = matched_output.get('decomposed2text', {})
    if decomposed2text:
      for element, score in decomposed2text.values():
        ent2score[element] = max(ent2score.get(element, 0), score)
    cropped2text = matched_output.get('cropped2text', {})
    if cropped2text:
      for element, score, coord in cropped2text.values():
        ent2score[element] = max(ent2score.get(element, 0), score)

    # create some qa from coordinates of elements     
    if cropped2text:
      background_element = None
      prev_small_element = None
      for element, score, coord in cropped2text.values():
        if score >= score_cutoff:
          if coord[0] <= 15 and coord[1] <= 15 and  coord[2] >= 85 and coord[3] >= 40:
            matched_output['qa'] = matched_output.get('qa',[]) +  [(element, f"what is in the background?|| {element}")] 
            background_element = element
            continue
          if (coord[2] - coord[0] <= 25 or coord[3] - coord[1] <= 25) and prev_small_element:
            prev_element, prev_score, prev_coord = prev_small_element
            if coord[0] - prev_coord[0] > 25: 
              if random.randint(0,1) == 0:
                matched_output['qa'] = matched_output.get('qa',[]) +  [(element +" and "+ prev_element, f"where is {prev_element} in relation to {element}?|| left")] 
              else:         
                matched_output['qa'] = matched_output.get('qa',[]) +  [(element +" and "+ prev_element, f"where is {element} in relation to {prev_element}?|| right")] 
              prev_small_element = None
              continue
            elif coord[2] - prev_coord[2] > 25:  
              if random.randint(0,1) == 0:
                matched_output['qa'] = matched_output.get('qa',[]) +  [(element +" and "+ prev_element, f"where is {prev_element} in relation to {element}?|| above")] 
              else:         
                matched_output['qa'] = matched_output.get('qa',[]) +  [(element +" and "+ prev_element, f"where is {element} in relation to {prev_element}?|| below")] 
              prev_small_element = None
              continue            
          if (coord[2] - coord[0] <= 50 or coord[3] - coord[1] <= 50):
            if  background_element:
              if random.randint(0,1) == 0:
                matched_output['qa'] = matched_output.get('qa',[]) +  [(element +" and "+ background_element, f"where is {background_element} in relation to {element}?|| behind")] 
              else:         
                matched_output['qa'] = matched_output.get('qa',[]) +  [(element +" and "+ background_element, f"where is {element} in relation to {background_element}?|| in front")] 
              background_element= None
            if coord[0] < 25:
              matched_output['qa'] = matched_output.get('qa',[]) +  [(element, f"where is {element}?|| left")]
            elif coord[0] > 75:
              matched_output['qa'] = matched_output.get('qa',[]) +  [(element, f"where is {element}?|| right")]
            elif coord[0] > 40 and coord[0] < 60 and coord[1] > 40 and coord[1] < 60:
              matched_output['qa'] = matched_output.get('qa',[]) +  [(element, f"where is {element}?|| center")]
            elif coord[1] < 25:
              matched_output['qa'] = matched_output.get('qa',[]) +  [(element, f"where is {element}?|| above")] 
            elif coord[1] > 75:
              matched_output['qa'] = matched_output.get('qa',[]) +  [(element, f"where is {element}?|| below")]
            if coord[2] - coord[0] <= 50 or coord[3] - coord[1] <= 25: prev_small_element = (element, score, coord)
            continue  
          
  # add some pre-created qa's
  for entity, question in potential_qa_list:
    print ('implied entity score', entity, ent2score.get(entity,0))
    if ent2score.get(entity,0) >= score_cutoff:
      answer = question.split("||")[-1].strip()
      if ent2score.get(answer,1) >= score_cutoff*high_score_mult:
          print ('implied answer score', answer, ent2score.get(answer,1))
          matched_output['qa'] = matched_output.get('qa',[]) +  [(entity, question)]
    elif " and " in entity: 
        entity1, entity2 = entity.split(" and ", 1)
        entity1, entity2 = entity1.strip(), entity2.strip()
        print ('implied entity1 score', entity1, ent2score.get(entity1,0))
        print ('implied entity2 score', entity2, ent2score.get(entity2,0))
        if (ent2score.get(entity1,0) >= score_cutoff) and (ent2score.get(entity2,0) >= score_cutoff):
          answer = question.split("||")[-1].strip()
          if ent2score.get(answer,1) >= score_cutoff*high_score_mult:
              matched_output['qa'] = matched_output.get('qa',[]) +  [(entity, question)]
        
      
#saves away a json of from {'matched_sentence': <str>, 'score': <float>, ...
#decomposed_image_features is shape=[1, 50, 512] dtype="float16"
#image is shape = [100,100,3], dtype="uint8"
#tokens is [1, 1028] int16
      
def create_synthetic_text_image_data(output_append_to_file, input_en_txt_gz_file, max_items=10000, score_cutoff=0.20, max_img_per_doc=5, trimmed_text_word_len=50, verbose=False, pytorch_device='cuda', high_score_mult=1.2):
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
        if l_lower.count("free") +  l_lower.count("viagra") + l_lower.count("cialis") > 3: continue 
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
        if "?||" in l:
          l, answer = l.split("?||",1)
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
        l, aug2ent, qa_list  = augment_ents(l, do_person=True, do_loc=False, do_obj=False, other_person_list=other_person_list)
        #if qa: qa_list.append(qa)
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
              text2 = trimmed_text
              text2_arr = text2.replace("?", ". ").replace("!", ". ").replace("- ", ". ").replace(";", ". ").strip(".").replace("..", ".").replace("..", ".").replace("..", ".").replace("  ", " ").split(". ")
              text3 = [t.strip() for t in  text2_arr if len(t.strip()) > 20]
              if len(text2_arr) > 1: text3 = text3 + [text2]
              # find the sentence that is most like this picture, the last item being the complete text chunk
              sim1 = 0.0
              clip_output = clip_image_to_multitext_score(clip_model, clip_processor, img, text3)
              if clip_output is None: continue
              most_similar_idx = clip_output['scores'].sort().indices[-1]
              sim1 = clip_output['scores'][most_similar_idx].item()
              # clip scores of generated images tend to be lower; this filters out really bad matches
              if sim1 >= score_cutoff:    
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
                # create some distractor phrases
                distractors=([] if 'eye' in matched_sentence else ['a closeup of an eye']) + ([] if 'face' in matched_sentence else ['a closeup of a face']) + ([] if 'network' in matched_sentence else ['diagram of lines and networks']) + ([] if ' clock ' in matched_sentence else ['clock']) + ([] if 'abstract' in matched_sentence else ['abstract art'])
                # infer implied entities based on the image
                potential_qa_list = create_qa_from_vlt5(matched_sentence, img,  aug2ent)
                implied_entities = [a[1].split("||")[1].strip() for a in potential_qa_list] 
                implied_entities = [a for a in implied_entities if a not in matched_sentence and a not in color_adj_set and a not in common_vlt5_words]
                print ('implied entities', implied_entities)
                potential_qa_list = list(set(potential_qa_list + qa_list))
                # now find the entities and important verbs in the most similar sentence.
                matched_output, cropped_images = get_sent_to_img(matched_sentence, img, get_cropped_images=True, other_sent_arr=distractors + implied_entities)
                distractors= set(distractors)
                distractor_is_best_match = False
                if matched_output:
                  matched_output['score'] = sim1
                if matched_output and matched_output['decomposed2text']:
                    items = list(matched_output['decomposed2text'].values())
                    items.sort(key=lambda a: a[1])
                    if items[-1][0] in distractors:
                       distractor_is_best_match = True
                if not matched_output or  distractor_is_best_match or matched_output['score'] < score_cutoff or \
                      (matched_output['decomposed2text'] and not any(a for a in matched_output['decomposed2text'].values() if a[1] >= score_cutoff)):
                    #this is an undrawable sentence
                    matched_output = {}
                    matched_output['matched_sentence'] = matched_sentence
                    matched_output['next_text'] = next_text
                    matched_output['prev_text'] = prev_text
                    if qa: matched_output['qa'] = matched_output.get('qa',[]) + [qa]
                    matched_output['qa'] = list(set(matched_output.get('qa',[])))
                    out.write(str(matched_output)+"\n")
                    continue
                else:
                    matched_output['tokens'] = tokens.tostring()
                    matched_output['thumbnail'] = np.array(img).tostring()
                    matched_output['prev_text'] = prev_text
                    matched_output['next_text'] = next_text
                    if qa: matched_output['qa'] = list(set(matched_output.get('qa',[]) + [qa]))
                    matched_output['qa'] = list(set(matched_output.get('qa',[])))
                    if matched_output['decomposed2text']: matched_output['decomposed2text'] = dict([(a, b) for a,b in matched_output['decomposed2text'].items() if b[0] not in distractors and not (b[0] in implied_entities and b[1] < score_cutoff*high_score_mult)])
                    if matched_output['cropped2text']: matched_output['cropped2text'] = dict([(a, b) for a,b in matched_output['cropped2text'].items() if b[0] not in distractors and not (b[0] in implied_entities and b[1] < score_cutoff*high_score_mult)])
                    create_qa(matched_output, img, score_cutoff, potential_qa_list=potential_qa_list, high_score_mult=high_score_mult)
                         
                    if verbose:
                      cropped2text = matched_output['cropped2text']
                      if cropped2text:
                        for idx, vals in cropped2text.items():
                          ci = cropped_images[idx]
                          print (vals)
                          if in_notebook: display(PIL.Image.fromarray(ci))
                      print ( matched_output['score'], '**', matched_output['matched_sentence'], '***', aug2ent, '***', matched_output['decomposed2text'], '***', matched_output.get('qa'))
                      if in_notebook: display(img)
                    dat_cnt += 1    
                    
                    #now let's create a different sentence based on the elements of the previous sentence, using words that have higher visual scores
                    new_words = [a[0] for a in  matched_output['decomposed2text'].values() if a[1] >= score_cutoff and a[0] in matched_sentence] 
                    word_str = ", ".join(new_words)
                    if word_str:
                      generated_sentence = commongen_model.generate(commongen_tokenizer.encode(word_str, return_tensors="pt").to(device), 
                                                                    min_length=len(word_str.split())*3, 
                                                                    max_length=len(word_str.split())*10, 
                                                                    no_repeat_ngram_size=2)
                      generated_sentence = commongen_tokenizer.decode(generated_sentence[0], skip_special_tokens=True).strip(". ")
                      if ".." in generated_sentence: generated_sentence, _ = generated_sentence.split("..", 1)
                      generated_sentence = generated_sentence.strip()
                      l_lower = generated_sentence.lower()
                      if l_lower.count(" sex ") + l_lower.count(" fuck ") + l_lower.count(" cock ") + l_lower.count(" pussy ") + l_lower.count(" xxx ") > 1: continue  
                      if "," in generated_sentence and generated_sentence.count(",") > len(generated_sentence.split())*.5: continue
                      orig_generated_sentence = generated_sentence
                      
                      #augment the sentence with fake data
                      mood_type = random.choice(["", "",  "", "", "",  "", ] + mood_lst)
                      image_type = random.choice(["", "",  "", "", "",  "",] + image_type_lst)
                      if not ("rendering" in image_type or "art" in image_type or "cartoon" in image_type or "illustration" in image_type or "drawing" in image_type or "sketch" in image_type):
                        mult = 1.0
                        prob_add_qa_image_type = 0.5
                        generated_sentence, aug2ent_gen, qa_list_gen  = augment_ents(generated_sentence, do_person=False, do_loc=True, do_obj=True, other_person_list=other_person_list)
                        generated_sentence = re_augment(generated_sentence, aug2ent) # put back in the augmented data from the original sentence
                        aug2ent_gen = dict(list(aug2ent_gen.items()) + list(aug2ent.items()))
                        qa_list_gen = qa_list_gen + qa_list
                      else:
                        #drawings can be more unrealistic so we want a higher match, and we don't further augment the sentence to improve the match
                        mult = high_score_mult
                        prob_add_qa_image_type = 1.0
                        qa_list_gen = qa_list
                        aug2ent_gen = aug2ent
                        
                      prefix = ""
                      if mood_type and not image_type:
                        prefix =  mood_type + " picture of:" 
                      elif not mood_type and image_type:
                        prefix = image_type +" of:"
                      elif mood_type and image_type:
                        prefix = mood_type + " " + image_type +" of:"
                      #generate an image 
                      tokens, img = minidalle.generate(prefix + " " + generated_sentence  if prefix else generated_sentence, image_output=True, token_output=True)
                      img = img.resize((100,100))
                      tokens = tokens.cpu().numpy()
                      tokens.dtype = np.int16
                      clip_output = clip_image_to_multitext_score(clip_model, clip_processor, img, [generated_sentence])
                      if clip_output is not None and clip_output['scores'][0] >= score_cutoff:
                        sim2 = clip_output['scores'][0].item()
                        # we only use the fake data to generate the image. the text2img matching uses the simplified sentence.
                        generated_sentence = simplify_aug(generated_sentence, aug2ent_gen)
                        distractors=([] if 'eye' in generated_sentence else ['a closeup of an eye']) + ([] if 'face' in generated_sentence else ['a closeup of a face']) + ([] if 'network' in generated_sentence else ['diagram of lines and networks']) + ([] if 'clock' in generated_sentence else ['clock']) + ([] if 'abstract' in generated_sentence else ['abstract art'])
                        potential_qa_list = create_qa_from_vlt5(generated_sentence, img,  aug2ent_gen)
                        implied_entities = [a[1].split("||")[1].strip() for a in potential_qa_list] + [a[0] for a in potential_qa_list] 
                        implied_entities = [a for a in implied_entities if a not in generated_sentence and a not in color_adj_set and a not in common_vlt5_words]
                        prefix_arr = []
                        if prefix:
                          prefix = prefix.replace(' of:', '')
                          prefix = prefix.split()
                          for pi in range(len(prefix)):
                            pr = " ".join(prefix[-pi+1:])
                            if pr not in generated_sentence:
                              implied_entities.append(pr)
                              prefix_arr.append(pr)
                        for word in new_words:
                          if word not in generated_sentence:
                            implied_entities.append(word)
                        implied_entities = list(set(implied_entities)) 
                        print ('implied entities', implied_entities)
                        potential_qa_list = potential_qa_list + qa_list_gen
                        matched_output2, cropped_images = get_sent_to_img(generated_sentence, img, get_cropped_images=True, 
                                                                          other_sent_arr=distractors + \
                                                                          implied_entities)
                        distractors = set(distractors)
                        distractor_is_best_match = False
                        if matched_output2:
                            matched_output2['score'] = sim2
                        if matched_output2 and matched_output2['decomposed2text'] and matched_output2['cropped2text']:
                            items = list(matched_output2['decomposed2text'].values())
                            items.sort(key=lambda a: a[1])
                            if items[-1][0] in distractors:
                              distractor_is_best_match = True
                              print ('distractor 1', items)
                        if matched_output2 and not distractor_is_best_match and \
                            matched_output2['decomposed2text'] and \
                            matched_output2['score'] >= mult*score_cutoff and \
                            len([a for a in matched_output2['decomposed2text'].values() if a[1] >= score_cutoff]) >= (len(matched_output2['decomposed2text'])*.5):
                          if matched_output2['decomposed2text']: matched_output2['decomposed2text'] = dict([(a, b) for a,b in matched_output2['decomposed2text'].items() if b[0] not in distractors and not (b[0] in implied_entities and b[1] < score_cutoff*high_score_mult)])
                          if matched_output2['cropped2text']: matched_output2['cropped2text'] = dict([(a, b) for a,b in matched_output2['cropped2text'].items() if b[0] not in distractors and not (b[0] in implied_entities and b[1] < score_cutoff*high_score_mult)])
                          create_qa(matched_output2, img, score_cutoff, potential_qa_list=potential_qa_list, high_score_mult=high_score_mult)
                          if  matched_output2['decomposed2text']:
                              matched_prefix = [a for a in matched_output2['decomposed2text'].values() if a[0] in prefix_arr] 
                              matched_prefix.sort(key=lambda a: len(a), reverse=True)
                              if not matched_prefix:
                                prefix = mood_type = image_type = None
                              else:
                                matched_prefix = matched_prefix[0]
                                matched_prefix = matched_prefix.split()
                                if len(matched_prefix) == 1:
                                  mood_type = None
                                  image_type = matched_prefix[0]
                                else:
                                  mood_type = matched_prefix[0]
                                  image_type = matched_prefix[1]
                            
                          if mood_type:
                            matched_output2['qa'] = list(set(matched_output2.get('qa',[]) + [('mood type', f'what is the mood of this picture?||{mood_type}')]))
                          if random.random() <= prob_add_qa_image_type:
                            if image_type== "": 
                              image_type = "photo"
                            if image_type is not None:
                              matched_output2['qa'] = list(set(matched_output2.get('qa',[]) + [('picture type', f'what type of picture is this?||{image_type}')]))
                          matched_output['tokens2'] = tokens.tostring()
                          matched_output['thumbnail2'] = np.array(img).tostring()
                          matched_output['score2'] = sim2
                          #matched_output['text2image_scores2'] = matched_output2['text2image_scores']
                          matched_output['decomposed2text2'] = matched_output2['decomposed2text']
                          matched_output['decomposed_image_features2'] = matched_output2['decomposed_image_features']
                          matched_output['cropped2text2'] = matched_output2['cropped2text']
                          matched_output['cropped_image_features2'] = matched_output2['cropped_image_features']
                          matched_output['image_features2'] = matched_output2['image_features']
                          matched_output['matched_sentence2'] = matched_output2['matched_sentence']
                          matched_output['qa2'] = list(set(matched_output2.get('qa',[])))
                          if verbose:
                            cropped2text = matched_output2['cropped2text']
                            if cropped2text:
                              for idx, vals in cropped2text.items():
                                ci = cropped_images[idx]
                                print (vals)
                                if in_notebook: display(PIL.Image.fromarray(ci))
                              print ('generated:', matched_output2['score'], '***'. prefix, '***', matched_output2['matched_sentence'],  '***', aug2ent_gen, '***', matched_output2['decomposed2text'], '***', matched_output2.get('qa'))
                              if in_notebook: display(img)  
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
    parser.add_argument('-verbose', dest='verbose', type=int, help='verbse mode', default= 0)
    parser.add_argument('-high_score_mult', dest='high_score_mult', type=float, help='multiple of score_cutff for implied data', default= 1.2)
    args = parser.parse_args()
    create_synthetic_text_image_data(output_append_to_file=args.output_append_to_file, \
                                     input_en_txt_gz_file=args.input_en_txt_gz_file, \
                                     max_items=args.max_items, \
                                     score_cutoff=args.score_cutoff, \
                                     max_img_per_doc=args.max_img_per_doc, \
                                     trimmed_text_word_len=args.trimmed_text_word_len, \
                                     verbose=args.verbose, high_score_mult=args.high_score_mult, \
                                     pytorch_device=args.pytorch_device)
 
