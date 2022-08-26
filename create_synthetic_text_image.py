import random
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
shape_adj = ["near", "far", "large", "small", "medium", "tall", "broad", "crooked", \
             "curved", "deep", "even", "flat", "hilly", "jagged", \
              "round", "shallow", "square", "steep", "straight", "thick", \
              "thin", "triangular", "uneven"]
color_adj = ["brown", "black", "blue", "gray", "green", \
             "pink", "purple", "red", "orange", "white", "yellow"]
#TODO improve this with more variety

def aug_obj(obj_str):
  adj =  random.choice(["", "", "", "", ]+shape_adj) + " "
  adj += " "+random.choice(["", "", "", "", ]+color_adj) + " "
  obj = " the " + adj
  obj =  obj.replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
  return obj

def aug_loc(loc_str=""):
  if loc_str and random.randint(0,1) == 0:
    loc =  " the " +random.choice(["", "", "", "", ]+shape_adj) + " "
  else:
    loc = " the " +random.choice(["", "", "", "", ]+shape_adj) + " " + random.choice(["place", "location", "locale", "site",]) + " "
  loc =  loc.replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
  return loc

def aug_person(person_str="", is_male=True):
  norp = ""
  norp += " " +random.choice(["", "", "", "", "gay", "straight", "bisexual",])
  norp += " " +random.choice(["", "", "", ""] + emotion_adj)
  norp += " " +random.choice(["", "", "", "", "conservative", "liberal", "moderate"])
  norp += " " +random.choice(["", "", "", "", "christian", "muslim", "buddhist", "hindu", ])
  norp += " " +random.choice(["", "", "", "", "white", "black", "asian", "middle-eastern", "african", "hispanic", "native", "indian"])
  norp += " " +random.choice(["", "", "", "", "young", "middle-aged", "old"])
  if person_str and random.randint(0,1) == 0: 
    person = "the " + norp + " " person_str
  else:
    if is_male: 
      person = "the " + norp + " " + random.choice(["man", "man", "man", "guy", "boy", "dude", "person"])
    else:
      person = "the " +  norp + " " + random.choice(["woman", "woman", "woman", "gal", "girl", "person"])
  person =  person.replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
  return person

def simplify_aug(sentence, all_aug):
  for el2 in all_aug:
    el2_arr = el2.split()
    if len(el2_arr) > 3:
      el2_arr = [el2_arr[0]] + el2_arr[-2:]
    el3 = " ".join(el2_arr)
    sentence = sentence.replace(el2, el3)
  return text


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

def get_decomposed_sent_to_img(matched_sentence, img, other_sent_arr=[]):
  global spacy_nlp, clip_model, clip_processor, minidalle, device, commongen_model, commongen_tokenizer
  doc = spacy_nlp(matched_sentence)
  noun_chunks = [e.text.replace("the ", "").replace("these ", "").replace("this ", "").replace("that ", "") for e in doc.noun_chunks]
  ner_and_verbs = dict([(e.text.lower() if len(e.text) < 5 else e.text.lower()[:5], e.text) for e in doc.ents if len(e.text) > 4] + \
                           [(e.text.lower() if len(e.text) < 5 else e.text.lower()[:5], e.text) for e in doc if len(e.text) > 4 and e.tag_.startswith('VB') and e.text.lower() not in stopwords_set] + \
                           [(e.lower() if len(e) < 5 else e.lower()[:5], e) for e in noun_chunks if len(e) > 4]) 
  text4 = list(set(list(ner_and_verbs.values()))) + other_sent_arr
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
      #image is shape = [75,75,3], dtype="uint8"
      #tokens is [1, 1028] int16
      most_similar_idx = clip_output['scores'].sort().indices[-1]
      sim1 = clip_output['scores'][most_similar_idx].item()
      matched_output = {'score': sim1, 'matched_sentence': matched_sentence, 'element2text': clip_output['element2text'], \
                              'decomposed_image_features': clip_output['decomposed_image_features'].cpu().numpy().tostring(), }
      return matched_output
  return None                    

def create_qa_vlt5(matched_output, img):
  global vlt5, vlt5_tokenizer
  
  element2text = matched_output['matched_output']
  pre_element = ""
  verb = ""
  for element, score in element2text
    if not (element.endswith("ed") or element.endswith("es") or element.endswith("ing")):
      if random.randint(0,1) == 0 and prev_element:
        answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: where is the {element}?",  img, 
                                        no_repeat_ngram_size=2, max_detections=5)
        matched_output['qa'] = matched_output.get('qa',[]) +  [f"where is the {element}?|| {answer}"]    
       else:
        answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: where is the {element} in relation to {prev_element}?",  img, 
                                        no_repeat_ngram_size=2, max_detections=5)
        matched_output['qa'] = matched_output.get('qa',[]) +  [f"where is the {element} in relation to {prev_element}?|| {answer}"]    
      if verb:
        answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: why is the {element} doing {verb}?",  img, 
                                        no_repeat_ngram_size=2, max_detections=5)  
        matched_output['qa'] = matched_output.get('qa',[]) +  [f"why is the {element} doing {verb}?|| {answer}"]                         
        answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what color is the {element}?",  img, 
                                        no_repeat_ngram_size=2, max_detections=5)
        matched_output['qa'] = matched_output.get('qa',[]) +  [f"vqa: what color is the {element}?|| {answer}"]  
        answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what size is the {element}?",  img, 
                                        no_repeat_ngram_size=2, max_detections=5)
        matched_output['qa'] = matched_output.get('qa',[]) +  [f"what size is the {element}?|| {answer}"]  
                                              out.write(str(matched_output)+"\n")  
        answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: what the {element} feeling?",  img, 
                                        no_repeat_ngram_size=2, max_detections=5)
        matched_output['qa'] = matched_output.get('qa',[]) +  [f"what is the {element} feeling?|| {answer}"]  
                                              out.write(str(matched_output)+"\n")  
        answer = vlt5_image2text(vlt5, vlt5_tokenizer, f"vqa: why is the {element} feeling {feeling}?",  img, 
                                        no_repeat_ngram_size=2, max_detections=5)
        matched_output['qa'] = matched_output.get('qa',[]) +  [f"why is the {element} feeling {feeling}?|| {answer}"]  
                                              out.write(str(matched_output)+"\n")  
        pre_element = element
    else:
      verb = element
      
def create_synthetic_text_image_data(output_append_to_file, input_en_txt_gz_file, max_items=10000, score_cutoff=0.20, max_img_per_doc=5, trimmed_text_word_len=50, verbose=False, pytorch_device='cuda'):
  global spacy_nlp, clip_model, clip_processor, minidalle, device, commongen_model, commongen_tokenizer
  init_data(input_en_txt_gz_file, pytorch_device=pytorch_device)
  with open(output_append_to_file, "a+") as out:
   with IndexedGzipFileExt("en.txt.gz") as f:
      for cnt in tqdm.tqdm(range(max_items)):
        j = random.randint(0, len(f)-1)
        l = f[j]
        l = l.decode().strip()
        qa = ""
        if "? ||" in l:
          l, answer = l.split("? ||",1)
          l_arr = l.split(".")
          l = ".".join(l_arr[:-1])
          qa = l_arr[-1]+"?|| "+answer
        if "You are visiting the placeholder page" in l: continue
        l_lower = l.lower()
        if l_lower.count("free") + l_lower.count("dating") + l_lower.count("sex") + l_lower.count("fuck") + l_lower.count("cock") + l_lower.count("pussy") + l_lower.count("xxx") > 3: continue  
        
        if l_lower.count("free") +  l_lower.count("viagra") +   l_lower.count("cialis") > 3: continue 
        l = l.replace("because this is pdf file * PDF *", " ... ")
        l = l.replace("no short description", " ... ")
        trimmed_text = l.lower().split()
        stopword_cnt = len([word for word in trimmed_text if word in stopwords_set ])
        shortword_cnt = len([word for word in trimmed_text if len(word) < 5])
        stopword_ratio = stopword_cnt/len(trimmed_text)
        shortword_ratio = shortword_cnt/len(trimmed_text)
        if not (stopword_ratio < 0.75 and random.random() < stopword_ratio and random.random() < shortword_ratio):
          continue
        doc = spacy_nlp(l)
        #TODO, detect public figures which we will replace with lower frequency
        person_ner =  list(set([(e.text, e.label_) for e in doc.ents if e.label_ == 'PERSON']))
        #print (person_ner)
        
        #gender swap to balance out the dataset
        if ("men " in l or " Men " in l or "men's " in l or " Men's " in l) and random.randint(0, 2) == 0:
          l = l.replace("men ", " women ").replace(" Men ", " Women ").replace("men's ", "women's ").replace(" Men's ", " Women's ")
        if ("man " in l or " Man " in l or "man's " in l or " Man's " in l) and random.randint(0, 2) == 0:
          l = l.replace("man ", "woman ").replace(" Man ", " Woman ").replace("man's ", "woman's ").replace(" Man's ", " Woman's ")
        if (" boys " in l or " Boys " in l or " boys' " in l or " Boy's " in l) and random.randint(0, 2) == 0:
          l = l.replace(" boys ", " girls ").replace(" Boys ", " Girls ").replace(" boys' ", " girls' ").replace(" Boys' ", " Girls' ")
        if (" boy " in l or " Boy " in l or " boy's " in l or " Boy's " in l) and random.randint(0, 2) == 0:
          l = l.replace(" boy ", " girl ").replace(" Boy ", " Girl ").replace(" boy's ", " girl's ").replace(" Boy's ", " Girl's ")
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
        person2person = {}
        if person_ner:
          for person_name, _ in person_ner:
            if random.randint(0,2)!=0:
              continue
            if not person:
              person = aug_person(person_str=person_name, is_male=random.randint(0,1)==0)
              continue
            if person_name.endswith("'s"): person_name = person_name[:-2]
            #print ("replacing **", person_name, person)
            person2person[person_name] = person
            l = l.replace(person_name, person, 1)
            person = aug_person(random.randint(0,1)==0)
        person = aug_person(person_str="", is_male=" he " in l or " He " in l)
        l = l.replace("Dr. the", "the").replace("Mr. the", "the").replace("Mrs. the", "the").replace("Miss. the", "the").replace("Ms. the", "the")
        l = l.replace("Dr the", "the").replace("Mr the", "the").replace("Mrs the", "the").replace("Miss the", "the").replace("Ms the", "the")
        #print (l)
        #l = l.split()
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
                                    no_repeat_ngram_size=2, max_detections=5)
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
                  prev_text = simplify_aug((prev_text2+". "+prev_text).strip(" ."), [person]+ list(person2person.values()))
                else:
                  prev_text = simplify_aug((prev_text2+" "+prev_text).strip(" ."), [person]+ list(person2person.values()))
                  next_text2 = "" if didx >= len(dat) -1 else  dat[didx+1]
                  if next_text2 and next_text and next_text2[0] == next_text2[0].upper():
                    next_text = simplify_aug((next_text +". "+next_text2).strip(" ."), [person]+ list(person2person.values()))
                  else:
                    next_text = simplify_aug((next_text +" "+ next_text2).strip(" ."), [person]+ list(person2person.values()))  
                
                #let's do some cleanup of the person mention since we injected more information then is in natural text
                matched_sentence = simplify_aug(matched_sentence, [person]+ list(person2person.values()))
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
                    do_more_vlt5 = False
                    vlt5_caption_with_score = [e for e in element2text.values() if e[0] == vlt5_caption]
                    if vlt5_caption_with_score:
                      if vlt5_caption_with_score[0][1] > 0.21:
                        matched_output['qa'] = matched_output.get('qa',[]) +  [f"Is there {vlt5_caption_with_score[0][0]} in this picture? || Yes"]
                        do_more_vlt5 = True
                      elif random.randint(0,5)==0:
                        matched_output['qa'] = matched_output.get('qa',[]) +  [f"Is there {vlt5_caption_with_score[0][0]} in this picture? || No"]  
                        do_more_vlt5 = False
                    if do_more_vlt5:
                      create_qa_vlt5(matched_output, img)
                    if verbose:
                      print ( matched_output['score'], '**', matched_output['matched_sentence'], '***', matched_output['element2text'])
                      if 'annotated_image' in vlt5_output['frcnn_output']:
                        display(vlt5_output['frcnn_output']['annotated_image'])
                      else:
                        display(img)
                    dat_cnt += 1    
                    word_str = ", ".join([a[0] for a in element2text.values() if a[1] > score_cutoff and a[0] != vlt5_caption])
                    if word_str:
                      
                      generated_sentence = commongen_model.generate(commongen_tokenizer.encode(word_str, return_tensors="pt").to(device), 
                                                                    min_length=len(word_str.split())*3, 
                                                                    max_length=len(word_str.split())*10, 
                                                                    no_repeat_ngram_size=2, )
                      generated_sentence = commongen_tokenizer.decode(generated_sentence[0], skip_special_tokens=True).strip(". ")
                      if ".." in generated_sentence: generated_sentence, _ = generated_sentence.split("..", 1)
                      generated_sentence = generated_sentence.strip()
                      tokens, img = minidalle.generate(generated_sentence, image_output=True, token_output=True)
                      img = img.resize((100,100))
                      tokens = tokens.cpu().numpy()
                      tokens.dtype = np.int16
                      doc = spacy_nlp(generated_sentence)
                      aug_ner =  []
                      for e in doc.ents:
                        aug_word =  aug_person(e.text, random.randint(0,1)) if e.label_ == 'PERSON' else aug_loc(e.text) if e.label_ == 'LOC' else aug_obj(e.text)  if e.label_ not in ("CARDINAL", "DATE") else e.text
                        generated_sentence = matched_sentence.replace(e.text, aug_word)
                        aug_ner.append(aug_word)
                      generated_sentence = simplify_aug(generated_sentence, aug_ner)
                      matched_output2 = get_decomposed_sent_to_img(generated_sentence, img)
                      if matched_output and matched_output['score'] > score_cutoff:
                        matched_output["element2text2"] = matched_output2["element2text"]
                        matched_output['matched_sentence2'] = matched_output2["matched_output"]
                        matched_output['score2'] = matched_output2["score"]
                        matched_output['decomposed_image_features2'] = matched_output2["decomposed_image_features"]
                        matched_output['tokens2'] = tokens.tostring()
                        matched_output['thumbnail2'] = np.array(img).tostring()
                        if verbose:
                            print ( matched_output2['score'], '**', matched_output2['matched_sentence'], '***', matched_output2['element2text'])
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
 
