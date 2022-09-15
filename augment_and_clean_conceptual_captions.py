
from collections import Counter
import subprocess
from urllib.parse import unquote

from nltk import wordnet
from nltk.corpus import wordnet as wn
import itertools

import json

ignore = [
    'confession bear',
    'white circle on',
    'concept showing a',
    'flag button series',
    'typography lettering phrase',
    'article list list'
    'illustrated country shape',
    'conceptual of a',
    'on the theme',
    'a sign for',
    'banner for',
    'map of',
    'maps of',
    'flag of',
    'flags of',
    'icon of',
    'icons of',
    'logo of',
    'logos of',
    'wallpaper',
    'pattern',
    'wallpapers',
    'flags',
    'icons',
    'patterns',
    'text message',
    'text messages',
    'label',
    'labels',
    'stylish text',
    'stamps',
    'stamp',
    'logo'
    'logos'
    '3d',
    'shutterstock',
    'map showing'
]
pats = [
    'posing',
    'poses', 
    'pose',
    'a facial close up of',
    'collage from',
    'views of',
    'a cartoon illustration of',
    'a cartoon vector of',
    'in the world',
    'an illustration of',
    'a photograph of',
    'a photo of',
    'a picture of',
    'a drawing of',
    'a vector of',
    'a picture',
    'a drawing',
    'a vector',
    'cartoon illustration of',
    'cartoon vector of',
    'illustration of',
    'photograph of',
    'photo of',
    'picture of',
    'drawing of',
    'vector of',
    'cartoon illustrations',
    'cartoon vectors',
    'cartoons',
    'illustrations',
    'photographs',
    'photos',
    'pictures',
    'human language',
    'drawings',
    'vectors',
    'seamless patterns',
    'cartoon illustration',
    'cartoon vector',
    'cartoon',
    'illustration',
    'photograph',
    'photo',
    'picture',
    'drawing',
    'vector',
    'seamless pattern',
    'stamp or label',
    'sky map with the name of the stars and constellations astronomical symbol',
    'team of peoples',
    '3d rendering',
    'd rendering',
    'isolated',
    'an of a colourfully filled outline of',
    'depositphotos stock',
    'depositphotos',
    'a closeup of',
    'a close up of',
    'a closeup view of',
    'a close up view of',
    'closeup of',
    'close up of',
    'close - up of',
    'closeup view of',
    'close up view of',
    'close - up view of',
    'close - up',
    'vintage style',
    'a d map on blue water background of',
    'map of',
    'a rendered of',
    'a rendering of',
    'a poster style from',
    'a poster for',
    'a portrait of',
    'large fullheightview view to'  ,
    'large exterior view to' ,
    'large fromfaraway view to' ,
    'large exterior rear view of',
    'view over',
    'large exterior side view of',
    'view of',
    'view from across',
    'view down',
    'view from',
    'short term rentals' ,
    'luxury vacation rental',
    'flat style'  ,
    'description',
    'a 3d render',
    'time lapse footage of',
    'a d render',
    'portrait of',
    'black and white',
    'silhouette of',
    'a group of',
    'a set of',
    'hi res crop',
    'episode pictured',
    'a pair of',
    'detail of',
    'members of the',
    'a row of',
    'detail of',
    'hand drawn of',
    'large fullheightview the',
    'close up portrait',
    'a logo sign',
    'a black and white',
    'an aerial the',
    'silhouette of',
    'clip art of',
    'map of',
    'this is',
    '3d render of',
    'd render of',
    'a close up',
    'sketch of',
    'the interior of',
    'graphicstock of',
    'full length portrait',
    'looking up at',
    'general atmosphere at',
    'large exterior',
    '3d digital render',
    'd digital render',
    'side profile of',
    'interior of',
    'low angle',
    'file photo dated',
    'road sign used',
    'greeting card with',
    'abstract of',
    'studio shot of',
    'a monochrome character',
    'street scene in',
    'general view during',
    'general view of',
    'a banner for',
    'interior of',
    'assets elleuk com',
    'a silhouette of',
    'group of',
    'black silhouette of',
    'abstract background with',
    'close up shot',
    'what a shot',
    'some of the',
    'the facade of',
    'large group of',
    'high angle',
    'image galleryimage',
    'flag of',
    'cute of',
    'the silhouette of',
    'this is',
    'a background for',
    'a selection of',
    'digital of',
    'vacation rentals condo',
    'close up',
    'Large fullheightview',
    'Large exterior',
    'full length',
    'Large fullheightview view',
    'Image galleryImage Mandatory',
    'graphic of',
    'set of',
    'profile of',
    'view across',
    'a bunch of',
    'Short term rentals',
    'a variety of',
    'aerial the',
    'view on the',
    'watercolor of',
    'hand drawn',
    'sketch style of',
    'the picture shows',
    'a view across',
    'facade of a',
    'Large exterior view',
    'abstract colorful with',
    'a couple of',
    'a lot of',
    'arsenal manager arsene',
    'facade of',
    'horizontal wide angle',
    'looking down on',
    'the skyline of',
    'a display of',
    'flat of a',
    'a long exposure',
#    'one of the',
    'example of a',
    'image of a',
    'time lapse of',
    'an image of',
    'image may contain',
    'image of :',
    'an image of',
    'image of',
    'image result for',
    'geographical feature category',
    'an example of',
    'time - lapse',
    'aerial shot of',
    'shot of',
    'addition to',
    'part of',
    'person is',
    "here 's",
    'administrative france map',
    'property image #',
    'image of',
    "it 's",
    'view to',
    "an artist 's",
    'there is',
    'wide shot of',
    "there 's a",
    'a shot of',
    'take a look',
    'house rental -',
    'map showing the',
    'a look at',
    'slow motion shot',
    'flying over',
    'example of',
    'aerial footage of',
    'these are the',
    'look at the',
    'there is',
    'this file ,',
    'book cover of',
    'the exterior of',
    'here is a',
    'the inside of',
#    'one of my',
    'macro shot of',
#    'one of many',
    '4k time lapse',
    'filming location is',
    'it was a',
    'most of the',
    'painting artist ,',
    'shadow of a',
    'part of a',
    'apartment rental -',
    'a section of',
    'the beauty of',
    'a general the',
    'file - in',
    'footage of a',
    "there 's",
    "here 's",
    'many of',
    'panning shot of',
    'you can see',
    'a closer look',
    'still life with',
    'here is the',
    'outside of the',
    'image : a',
    'there are many',
    'welcome to the',
    'a number of',
    'the art of',
    'looking down the',
    'a member of',
    'natural beauty :',
    'looking at the',
    'file dated of',
    'this was the',
    'image of an',
    'hand - drawn',
    'video of a',
    'an aerial shot',
    'file - file',
    'outside of a',
    'save the date',
    'medium shot of',
    'there was a',
    'a wide shot',
    'this was taken',
    'an aerial view',
    'this map shows',
    'check out this',
    'all of the',
    'video footage of',
    'shot of the',
    'the cast of',
    'silhouettes of a',
    'the texture of',
    'this would be',
    'line of a',
#    'one of a',
    'a few of',
    'slow motion of',
    "the world 's",
    'design ideas for',
    '3d cg rendering',
    'high quality map',
    'here are the',
    'the most popular',
    'a seamless patterned',
    'looking for a',
    'type of place',
    'time for a',
    'i love this',
    'here are some',
    'this jpeg image',
    'seamless texture with',
    'a series of',
    'a guide to',
    'return to the',
    'villa rental -',
    'check out the',
    'visual artist ,',
#    'one of our',
    'general the atmosphere',
    'i want this', 
    'i want a',
    'i want an',
    'i would love',
    'i would like',
    'i like a',
    'i love a',
    'i really like a',
    'i really love a',
    'i like an',
    'i love an',
    'i really like an',
    'i really love an',
    'i like the',
    'i love the',
    'i really like the',
    'i really love the',
    'i like this',
    'i love this',
    'i really like this',
    'i really love this',
    "i 'm",
    'all about',
    'this image shows',
    'the logo is',
    'featured image for',
    'greeting card featuring',
    'i want to',
    'i have a',
    'type of',
    'a horizontal of',
    'a vertical of',
    '4k footage of',
    'flock of',
    'herd of',
    'image titled',
    'digital art selected', 
    'a painting of', 
    'this file ,', 
    'this file', 
    'one of my', 
    'the digital art', 
    'pictured',
    'a white background', 
    'against sports team', 
    'during the match', 
    'on white background', 
    'in the background', 
    '-- stock #', 
    'a black background', 
    'with clipping path', 
    'royalty - free', 
    'the white background', 
    'horizontal large gallery', 
    'by painting artist', 
    'in slow motion', 
    'a blue background', 
    'from the film', 
    'black and white', 
    'close - up', 
    'slow motion', 
    'live on stage', 
    'on a white', 
    'a dark background', 
    '- height view', 
    'on black background', 
    'on the background', 
    'at the camera', 
    'by film director', 
    'over white background', 
    'a green background', 
    'painting by person', 
    'for a portrait', 
    'on white', 
    'against white background', 
    #'background', 
    'a red background', 
    'background art illustration', 
    'in the film', 
    'for your design', 
    '- exterior photo', 
    'white background illustration', 
    'elements for your', 
    'background Stock Photo', 
    'a gray background', 
    'via sharing website', 
    ', by person', 
    'on the set', 
    'a wooden background', 
    ': stock illustration', 
    'for a photo', 
    'into the camera', 
    'looking at camera', 
    'the website homepage', 
    'free stock illustrations', 
    ', close up', 
    'with copy space', 
    'a pink background', 
    'the black background', 
    'the art illustration', 
    'a light background', 
    'as a background', 
    'of the movie', 
    'designed by person', 
    'towards the camera', 
    'background seamless pattern', 
    'the image link', 
    'oil on canvas', 
    'for the camera', 
    'background stock photo', 
    'on 500px',
    '3d',
    '4k',

] + ignore

pats.sort(key=lambda a: len(a), reverse=True)

plant_animal_descriptor = ("popular", "fallow", "nocturnal", "night", "invasive", "native", "foreign", "ancient", "extinct", "juvenile", "alpine", "river", "senior", "succulent", "tall", "dwarf", "giant", "herbal", "indoor", "outdoor", "island", "underwater", "grown", "growing", "lavendar", "resting", "rocky", "rock", "field", "garden", "beach", "immature", "tame", "zoo", "captive", "arctic", "jungle", "polar", "sand", "barn", "strange", "weird", "hairless", "golden", "happy", "sad", "marine", "medium", "miniature", "mom", "dad", "mother", "baby", "new", "old", "rare", "rock", "mountain", "magcal", "forest", "adorable", "pretty", "beautiful", "lush", "crawling", "shaded", "potted", "climbing", "blooming", "realistic", "winter", "summer", "autumn", "fall", "spring", "northern", "western", "eastern", "tattoo", "snow", "snowy", "full", "food", "house", "display", "large", "camouflage", "exotic", "solitary", "camouflaged", "southwestern", "northeastern", "forest", "early", "annual", "late", "flowering", "colorful", "cactus", "playful", "toy", "funny", "happy", "curious", "domestic", "farm", "friendly", "house", "live", "dead", "stuffed", "wild", "adult", "elderly", "fake", "tree", "fat", "skinny", "obese", "pedigree", "pedigreed", "pet", "rare", "land", "water", "ocean", "sea", "desert", "baby", "young", "old", "wild", "tropical", "spotted", "spotless", "sea", "white", "black", "yellow", "green", "blue", "orange", "purple", "red", "brown", "pink", "grey", "gray", "cute", "pretty", "beautiful", "male", "female", "juenile", "little", "small")
animal_list = {'dino': 1, 'leop': 1, 'baby': 1, 'swim': 1, 'turt': 1, 'frog': 1, 'fora': 1, 'hawk': 1, 'sava': 1, 'flie': 1, 'subs': 1, 'peli': 1, 'falc': 1, 'tedd': 1, 'biso': 1, 'jura': 1, 'game': 1, 'lynx': 1, 'geol': 1, 'gees': 1, 'tort': 1, 'butt': 1, 'seal': 1, 'catc': 1, 'arde': 1, 'eggs': 1, 'cret': 1, 'crui': 1, 'meal': 1, 'mudd': 1, 'cerv': 1, 'mara': 1, 'bots': 1, 'grum': 1, 'boat': 1, 'masa': 1, 'sciu': 1, 'robi': 1, 'pard': 1, 'kill': 1, 'scra': 1, 'body': 1, 'hone': 1, 'bull': 1, 'rave': 1, 'educ': 1, 'legs': 1, 'stuf': 1, 'elap': 1, 'hali': 1, 'rept': 1, 'keny': 1, 'chob': 1, 'resc': 1, 'groo': 1, 'tong': 1, 'kite': 1, 'dama': 1, 'koal': 1, 'talo': 1, 'floe': 1, 'soar': 1, 'zebr': 1, 'meer': 1, 'moos': 1, 'graz': 1, 'prem': 1, 'nami': 1, 'catt': 1, 'amur': 1, 'cubs': 1, 'mant': 1, 'peek': 1, 'calf': 1, 'trou': 1, 'pree': 1, 'laru': 1, 'otte': 1, 'neck': 1, 'mona': 1, 'ridg': 1, 'cats': 1, 'flap': 1, 'rein': 1, 'ante': 1, 'roar': 1, 'leve': 1, 'rant': 1, 'gira': 1, 'mule': 1, 'alpi': 1, 'pray': 1, 'gann': 1, 'tyto': 1, 'belg': 1, 'myda': 1, 'wool': 1, 'scar': 1, 'preh': 1, 'bobc': 1, 'veno': 1, 'pica': 1, 'carc': 1, 'guar': 1, 'bees': 1, 'toad': 1, 'humm': 1, 'chew': 1, 'arge': 1, 'bubo': 1, 'asle': 1, 'cham': 1, 'asho': 1, 'auro': 1, 'pole': 1, 'paus': 1, 'tenn': 1, 'tanz': 1, 'pige': 1, 'smil': 1, 'orca': 1, 'oryx': 1, 'dogs': 1, 'brou': 1, 'aggr': 1, 'loxo': 1, 'meat': 1, 'migr': 1, 'rufu': 1, 'duba': 1, 'alga': 1, 'prow': 1, 'pele': 1, 'lies': 1, 'tank': 1, 'hove': 1, 'gaze': 1, 'jell': 1, 'cock': 1, 'bigh': 1, 'scav': 1, 'neth': 1, 'draf': 1, 'clyd': 1, 'huds': 1, 'calm': 1, 'floc': 1, 'leap': 1, 'scor': 1, 'cric': 1, 'fjor': 1, 'rece': 1, 'raci': 1, 'ledg': 1, 'adel': 1, 'inne': 1, 'legg': 1, 'kang': 1, 'mane': 1, 'ibis': 1, 'nige': 1, 'capp': 1, 'corm': 1, 'antl': 1, 'nose': 1, 'seaw': 1, 'axis': 1, 'cind': 1, 'alio': 1, 'dean': 1, 'mans': 1, 'boob': 1, 'bait': 1, 'pouc': 1, 'sask': 1, 'peru': 1, 'bodi': 1, 'gami': 1, 'fera': 1, 'capu': 1, 'kori': 1, 'chas': 1, 'picn': 1, 'lays': 1, 'unci': 1, 'haul': 1, 'wale': 1, 'snac': 1, 'unic': 1, 'coel': 1, 'puma': 1, 'bann': 1, 'awar': 1, 'glac': 1, 'imma': 1, 'owne': 1, 'ears': 1, '500p': 1, 'died': 1, 'samb': 1, 'tigr': 1, 'quai': 1, 'peaf': 1, 'hood': 1, 'flea': 1, 'swai': 1, 'bron': 1, 'drak': 1, 'kest': 1, 'skie': 1, 'komo': 1, 'dieg': 1, 'eyed': 1, 'odoc': 1, 'armo': 1, 'bliz': 1, 'cows': 1, 'llam': 1, 'tuft': 1, 'lowl': 1, 'knit': 1, 'equu': 1, 'fine': 1, 'howl': 1, 'took': 1, 'ance': 1, 'rapt': 1, 'grif': 1, 'payn': 1, 'dock': 1, 'fact': 1, 'stur': 1, 'maga': 1, 'five': 1, 'clic': 1, 'matu': 1, 'coug': 1, 'deck': 1, 'samm': 1, 'babo': 1, 'pron': 1, 'delh': 1, 'narc': 1, 'scub': 1, 'mong': 1, 'soun': 1, 'agam': 1, 'tuat': 1, 'flyc': 1, 'maas': 1, 'gath': 1, 'exte': 1, 'loun': 1, 'lens': 1, 'tawn': 1, 'rutt': 1, 'mell': 1, 'rode': 1, 'bufo': 1, 'nove': 1, 'pusi': 1, 'falk': 1, 'skel': 1, 'scru': 1, 'etos': 1, 'laid': 1, 'wade': 1, 'plov': 1, 'dein': 1, 'pigl': 1, 'quay': 1, 'tale': 1, 'mome': 1, 'rack': 1, 'peck': 1, 'laug': 1, 'busi': 1, 'offs': 1, 'ants': 1, 'song': 1, 'rhea': 1, 'carv': 1, 'apis': 1, 'towe': 1, 'biti': 1, 'clon': 1, 'canv': 1, 'slid': 1, 'alpa': 1, 'ratt': 1, 'damp': 1, 'hyen': 1, 'kudu': 1, 'spit': 1, 'sval': 1, 'chih': 1, 'lift': 1, 'lark': 1, 'prob': 1, 'pups': 1, 'dalt': 1, 'subm': 1, 'anky': 1, 'toon': 1, 'hoar': 1, 'issa': 1, 'bair': 1, 'tapi': 1, 'clow': 1, 'dove': 1, 'kaya': 1, 'coot': 1, 'artw': 1, 'mend': 1, 'snar': 1, 'ibex': 1, 'rend': 1, 'dodo': 1, 'wido': 1, 'itch': 1, 'newt': 1, 'nilo': 1, 'estu': 1, 'kiss': 1, 'prom': 1, 'acry': 1, 'hyde': 1, 'surv': 1, 'sust': 1, 'snap': 1, 'days': 1, 'cert': 1, 'dalm': 1, 'rear': 1, 'newb': 1, 'less': 1, 'peer': 1, 'labr': 1, 'outc': 1, 'golf': 1, 'catf': 1, 'bone': 1, 'diet': 1, 'belu': 1, 'nova': 1, 'smok': 1, 'mill': 1, 'seag': 1, 'taro': 1, 'aflo': 1, 'fing': 1, 'ostr': 1, 'dena': 1, 'pied': 1, 'okav': 1, 'boze': 1, 'sauc': 1, 'foal': 1, 'jame': 1, 'mega': 1, 'nanc': 1, 'corb': 1, 'font': 1, 'dall': 1, 'sewa': 1, 'remo': 1, 'horr': 1, 'jett': 1, 'lace': 1, 'sigh': 1, 'kazi': 1, 'crot': 1, 'apat': 1, 'serp': 1, 'upli': 1, 'maje': 1, 'seat': 1, 'cere': 1, 'chro': 1, 'okla': 1, 'said': 1, 'loon': 1, 'quie': 1, 'fanc': 1, 'feel': 1, 'hemi': 1, 'bomb': 1, 'prev': 1, 'putt': 1, 'zanz': 1, 'paru': 1, 'than': 1, 'trek': 1, 'rana': 1, 'isle': 1, 'thru': 1, 'jasp': 1, 'avat': 1, 'leas': 1, 'dunn': 1, 'velo': 1, 'sous': 1, 'whoo': 1, 'verd': 1, 'titm': 1, 'adve': 1, 'phas': 1, 'pist': 1, 'crun': 1, 'enve': 1, 'emir': 1, 'curr': 1, 'pupp': 1, 'mugg': 1, 'phea': 1, 'elan': 1, 'joey': 1, 'tusk': 1, 'ovis': 1, 'pyrr': 1, 'pavo': 1, 'tool': 1, 'pika': 1, 'busy': 1, 'sung': 1, 'atra': 1, 'subf': 1, 'expl': 1, 'rump': 1, 'meme': 1, 'hide': 1, 'seab': 1, 'pock': 1, 'endi': 1, 'enco': 1, 'toco': 1, 'targ': 1, 'dwel': 1, 'katm': 1, 'skyl': 1, 'tigh': 1, 'notc': 1, 'guys': 1, 'satu': 1, 'morp': 1, 'slug': 1, 'case': 1, 'wart': 1, '1600': 1, 'dxov': 1, 'teto': 1, 'upon': 1, 'walr': 1, 'fawn': 1, 'lest': 1, 'sall': 1, 'fluf': 1, 'ador': 1, 'vaca': 1, 'alth': 1, 'loud': 1, 'jama': 1, 'icon': 1, 'disn': 1, 'naku': 1, 'inju': 1, 'ditc': 1, 'seni': 1, 'brau': 1, 'anis': 1, 'sill': 1, 'eggp': 1, 'scoo': 1, 'pref': 1, 'safa': 1, 'goli': 1, 'oils': 1, 'dung': 1, 'cloc': 1, 'gart': 1, 'rspb': 1, 'rudd': 1, 'taxi': 1, 'stak': 1, 'rufo': 1, 'safe': 1, 'sika': 1, 'hurr': 1, 'greb': 1, 'styr': 1, 'dilo': 1, 'vant': 1, 'guat': 1, 'bury': 1, 'maro': 1, 'inha': 1, 'seaf': 1, 'bloc': 1, 'diur': 1, 'amor': 1, 'buoy': 1, 'musc': 1, 'trin': 1, 'homo': 1, 'erec': 1, 'push': 1, 'cowb': 1, 'lock': 1, 'aver': 1, 'muni': 1, 'lonc': 1, 'glov': 1, 'sake': 1, 'file': 1, 'demo': 1, 'arri': 1, 'dory': 1, 'andr': 1, 'bate': 1, 'kaua': 1, 'fier': 1, 'toys': 1, 'clot': 1, 'drif': 1, 'oppo': 1, 'snat': 1, 'grus': 1, 'whos': 1, 'musi': 1, 'beca': 1, 'anza': 1, 'beam': 1, 'mast': 1, 'suck': 1, 'ladd': 1, 'argu': 1, 'gazi': 1, 'depi': 1, 'comf': 1, 'chit': 1, 'talk': 1, '135t': 1, 'lapl': 1, 'cita': 1, 'adop': 1, 'pang': 1, 'catb': 1, 'trog': 1, 'stuc': 1, 'ario': 1, 'vast': 1, 'welc': 1, 'bale': 1, 'abun': 1, 'wyom': 1, 'dini': 1, 'gila': 1, 'tint': 1, 'weas': 1, 'puts': 1, 'hike': 1, 'onco': 1, 'adva': 1, 'leig': 1, 'upse': 1, 'apes': 1, 'heap': 1, 'excl': 1, 'icef': 1, 'cous': 1, 'neph': 1, 'kana': 1, 'tops': 1, 'shre': 1, 'yeme': 1, 'nois': 1, 'pana': 1, 'desk': 1, 'esse': 1, 'arms': 1, 'sayi': 1, 'larc': 1, 'bact': 1, 'stif': 1, 'volu': 1, 'zsls': 1, 'ambu': 1, 'eiff': 1, 'nick': 1, 'wise': 1, 'trap': 1, 'else': 1, 'eith': 1, 'prud': 1, 'cozy': 1, 'algo': 1, 'voca': 1, 'lame': 1, 'prad': 1, 'lito': 1, 'kuya': 1, 'voiv': 1, 'jost': 1, 'thea': 1, 'noki': 1, 'hopi': 1, 'colt': 1, 'cygn': 1, 'rwan': 1, 'awai': 1, 'rota': 1, 'wres': 1, 'sail': 1, 'myan': 1, 'jers': 1, 'bhop': 1, 'logg': 1, 'gems': 1, 'fiji': 1, 'resp': 1, 'moat': 1, 'nene': 1, 'equi': 1, 'foxe': 1, 'angr': 1, 'loft': 1, 'zoos': 1, 'unna': 1, 'crou': 1, 'caim': 1, 'soci': 1, 'syri': 1, 'conu': 1, 'orni': 1, 'secl': 1, 'fuli': 1, 'kied': 1, 'spaw': 1, 'kiwi': 1, 'donk': 1, 'inle': 1, 'puer': 1, 'laun': 1, 'goof': 1, 'truc': 1, 'belt': 1, 'casu': 1, 'assa': 1, 'risi': 1, 'masc': 1, 'blee': 1, 'feas': 1, 'refe': 1, 'matc': 1, 'guja': 1, 'taxu': 1, 'bacc': 1, 'modi': 1, 'pemb': 1, 'molt': 1, 'phoe': 1, 'norv': 1, 'seah': 1, 'audu': 1, 'denv': 1, 'news': 1, 'espa': 1, 'pike': 1, 'grew': 1, 'swoo': 1, 'verc': 1, 'lurk': 1, 'trib': 1, 'garb': 1, 'gnaw': 1, 'duis': 1, 'sinc': 1, 'avda': 1, 'erkt': 1, 'olde': 1, 'scie': 1, 'acut': 1, 'bren': 1, 'agre': 1, 'bred': 1, 'eila': 1, 'mole': 1, 'waxw': 1, 'phal': 1, 'mull': 1, 'mehn': 1, 'weav': 1, 'saru': 1, 'wadd': 1, 'peni': 1, 'repo': 1, 'effi': 1, 'rall': 1, 'plaz': 1, 'myer': 1, 'nash': 1, 'fool': 1, 'bold': 1, 'wari': 1, 'lamp': 1, 'wher': 1, 'marn': 1, 'samp': 1, 'nook': 1, 'sled': 1, 'maur': 1, 'nutr': 1, 'upst': 1, 'urch': 1, 'brah': 1, 'exci': 1, 'meye': 1, 'vent': 1, 'fund': 1, 'dach': 1, 'anda': 1, 'ruff': 1, 'alre': 1, 'darw': 1, 'shet': 1, 'steg': 1, 'skua': 1, 'impe': 1, 'toro': 1, 'pops': 1, 'esca': 1, 'alda': 1, 'toed': 1, 'hugg': 1, 'cruz': 1, 'pseu': 1, 'amph': 1, 'proc': 1, 'magi': 1, 'wige': 1, 'warn': 1, 'kerm': 1, 'nott': 1, 'fail': 1, 'dati': 1, 'capi': 1, 'berk': 1, 'vita': 1, 'swed': 1, 'gulf': 1, 'hiki': 1, 'aven': 1, 'flar': 1, 'wels': 1, 'menu': 1, 'bert': 1, 'grac': 1, 'pata': 1, 'baja': 1, 'towp': 1, 'diep': 1, 'prou': 1, 'dubl': 1, 'toot': 1, 'wake': 1, 'dend': 1, 'naps': 1, 'trie': 1, 'unat': 1, 'lots': 1, 'snif': 1, 'vote': 1, 'unkn': 1, 'skyw': 1, 'delp': 1, 'lewa': 1, 'infa': 1, 'olym': 1, 'toss': 1, 'ecle': 1, 'spli': 1, 'bono': 1, 'iqui': 1, 'citi': 1, 'jaxc': 1, 'writ': 1, 'seei': 1, 'kona': 1, 'lori': 1, 'pood': 1, 'grab': 1, 'dump': 1, 'tugb': 1, 'hive': 1, 'deal': 1, 'scis': 1, 'ayor': 1, 'tana': 1, 'kanh': 1, 'fowl': 1, 'honk': 1, 'plei': 1, 'enti': 1, 'caut': 1, 'smit': 1, 'cico': 1, 'dale': 1, 'kunk': 1, 'exis': 1, 'butc': 1, 'trut': 1, 'perh': 1, 'obta': 1, 'vuln': 1, 'lege': 1, 'hidd': 1, 'loom': 1, 'ribe': 1, 'tick': 1, 'mott': 1, 'capa': 1, 'refu': 1, 'decl': 1, 'seek': 1, 'stub': 1, 'shep': 1, 'flee': 1, 'dayl': 1, 'hick': 1, 'bunt': 1, 'musk': 1, 'bass': 1, 'orla': 1, 'sabl': 1, 'halc': 1, 'devi': 1, 'ranc': 1, 'chen': 1, 'stou': 1, 'duke': 1, 'eagl': 1, 'anim': 1, 'reef': 1, 'bald': 1, 'flig': 1, 'mall': 1, 'pant': 1, 'pedi': 1, 'hare': 1, 'male': 1, 'squi': 1, 'perc': 1, 'lion': 1, 'pand': 1, 'ther': 1, 'flyi': 1, 'pose': 1, 'beng': 1, 'swan': 1, 'jung': 1, 'cape': 1, 'rhin': 1, 'exti': 1, 'prof': 1, 'dolp': 1, 'enjo': 1, 'burr': 1, 'capt': 1, 'pean': 1, 'prey': 1, 'play': 1, 'buff': 1, 'salm': 1, 'walk': 1, 'tige': 1, 'hero': 1, 'caug': 1, 'hunt': 1, 'gull': 1, 'liza': 1, 'suma': 1, 'igua': 1, 'maca': 1, 'reed': 1, 'sits': 1, 'thre': 1, 'afte': 1, 'arct': 1, 'goat': 1, 'duck': 1, 'move': 1, 'harb': 1, 'week': 1, 'tern': 1, 'ende': 1, 'cora': 1, 'wand': 1, 'eura': 1, 'gray': 1, 'leuc': 1, 'encl': 1, 'towa': 1, 'nest': 1, 'shor': 1, 'allo': 1, 'mute': 1, 'tund': 1, 'watc': 1, 'race': 1, 'huma': 1, 'jagu': 1, 'acti': 1, 'hipp': 1, 'monk': 1, 'stic': 1, 'kitt': 1, 'head': 1, 'afri': 1, 'ball': 1, 'prai': 1, 'tyra': 1, 'curi': 1, 'lepu': 1, 'comi': 1, 'shak': 1, 'reco': 1, 'stea': 1, 'coup': 1, 'peng': 1, 'bill': 1, 'barn': 1, 'stan': 1, 'rese': 1, 'entr': 1, 'acro': 1, 'herd': 1, 'pool': 1, 'tour': 1, 'hind': 1, 'adul': 1, 'real': 1, 'cute': 1, 'egre': 1, 'pier': 1, 'vers': 1, 'trip': 1, 'intr': 1, 'eati': 1, 'beac': 1, 'cold': 1, 'cons': 1, 'feed': 1, 'ferr': 1, 'chel': 1, 'must': 1, 'crow': 1, 'bala': 1, 'atte': 1, 'sitt': 1, 'fish': 1, 'chec': 1, 'cari': 1, 'hist': 1, 'mout': 1, 'rive': 1, 'alph': 1, 'bath': 1, 'aust': 1, 'clas': 1, 'conf': 1, 'spin': 1, 'tail': 1, 'deer': 1, 'arab': 1, 'stru': 1, 'claw': 1, 'chim': 1, 'bett': 1, 'mate': 1, 'sear': 1, 'wing': 1, 'seco': 1, 'kala': 1, 'term': 1, 'sunb': 1, 'crit': 1, 'exhi': 1, 'cate': 1, 'dang': 1, 'roos': 1, 'cris': 1, 'parr': 1, 'moti': 1, 'look': 1, 'stor': 1, 'mamm': 1, 'away': 1, 'gett': 1, 'live': 1, 'plai': 1, 'whil': 1, 'foot': 1, 'retu': 1, 'buzz': 1, 'racc': 1, 'alas': 1, 'inte': 1, 'croc': 1, 'serv': 1, 'pert': 1, 'horn': 1, 'amer': 1, 'shel': 1, 'feat': 1, 'clin': 1, 'bree': 1, 'crab': 1, 'waln': 1, 'hawa': 1, 'lond': 1, 'chic': 1, 'regi': 1, 'goos': 1, 'salt': 1, 'dist': 1, 'vult': 1, 'stri': 1, 'geor': 1, 'poss': 1, 'meet': 1, 'atta': 1, 'supe': 1, 'gala': 1, 'bott': 1, 'view': 1, 'figh': 1, 'snow': 1, 'wait': 1, 'bust': 1, 'cres': 1, 'pare': 1, 'beak': 1, 'rela': 1, 'magp': 1, 'sibe': 1, 'time': 1, 'wrap': 1, 'warm': 1, 'bear': 1, 'taki': 1, 'youn': 1, 'lake': 1, 'left': 1, 'nuts': 1, 'cana': 1, 'iceb': 1, 'dryi': 1, 'gall': 1, 'elep': 1, 'face': 1, 'layi': 1, 'inst': 1, 'blad': 1, 'stin': 1, 'disc': 1, 'danc': 1, 'pull': 1, 'apar': 1, 'post': 1, 'stoo': 1, 'weat': 1, 'spot': 1, 'vulp': 1, 'beav': 1, 'came': 1, 'happ': 1, 'ocea': 1, 'yawn': 1, 'mous': 1, 'harr': 1, 'carp': 1, 'appr': 1, 'aler': 1, 'sett': 1, 'alli': 1, 'pola': 1, 'juve': 1, 'trac': 1, 'dres': 1, 'divi': 1, 'euro': 1, 'chaf': 1, 'neig': 1, 'offe': 1, 'poun': 1, 'pome': 1, 'empe': 1, 'curl': 1, 'buck': 1, 'herr': 1, 'grey': 1, 'shee': 1, 'cast': 1, 'babi': 1, 'funn': 1, 'chee': 1, 'indi': 1, 'enda': 1, 'cros': 1, 'shar': 1, 'dead': 1, 'wate': 1, 'scro': 1, 'penc': 1, 'sere': 1, 'brin': 1, 'stat': 1, 'mosq': 1, 'spen': 1, 'ripp': 1, 'warb': 1, 'wadi': 1, 'spre': 1, 'runs': 1, 'bank': 1, 'cran': 1, 'braz': 1, 'diff': 1, 'mars': 1, 'alba': 1, 'bumb': 1, 'finc': 1, 'grav': 1, 'peri': 1, 'hors': 1, 'nile': 1, 'retr': 1, 'dusk': 1, 'angl': 1, 'glor': 1, 'barb': 1, 'nigr': 1, 'park': 1, 'foun': 1, 'swam': 1, 'coul': 1, 'ospr': 1, 'wash': 1, 'roll': 1, 'hidi': 1, 'muse': 1, 'timb': 1, 'ride': 1, 'glid': 1, 'gori': 1, 'sand': 1, 'fema': 1, 'lead': 1, 'four': 1, 'whal': 1, 'goes': 1, 'bask': 1, 'moun': 1, 'dive': 1, 'silh': 1, 'king': 1, 'brea': 1, 'brus': 1, 'repr': 1, 'bite': 1, 'impa': 1, 'poke': 1, 'esta': 1, 'rabb': 1, 'mala': 1, 'quic': 1, 'pond': 1, 'orde': 1, 'wolv': 1, 'jump': 1, 'chil': 1, 'coat': 1, 'open': 1, 'traf': 1, 'chai': 1, 'obse': 1, 'bigg': 1, 'anta': 1, 'rura': 1, 'driv': 1, 'snak': 1, 'mead': 1, 'wave': 1, 'posi': 1, 'marb': 1, 'worm': 1, 'smoo': 1, 'sour': 1, 'hamp': 1, 'spee': 1, 'snea': 1, 'city': 1, 'limb': 1, 'rest': 1, 'slee': 1, 'jack': 1, 'grou': 1, 'road': 1, 'onto': 1, 'hyac': 1, 'whol': 1, 'pres': 1, 'runn': 1, 'dome': 1, 'minn': 1, 'ship': 1, 'skin': 1, 'thom': 1, 'cust': 1, 'chur': 1, 'bush': 1, 'aqua': 1, 'scen': 1, 'seen': 1, 'rock': 1, 'neve': 1, 'doug': 1, 'bran': 1, 'coll': 1, 'redw': 1, 'plum': 1, 'work': 1, 'wins': 1, 'unde': 1, 'hump': 1, 'albe': 1, 'glau': 1, 'freq': 1, 'stee': 1, 'sanc': 1, 'ursu': 1, 'part': 1, 'nort': 1, 'star': 1, 'sign': 1, 'vanc': 1, 'tryi': 1, 'scho': 1, 'eare': 1, 'cent': 1, 'spec': 1, 'suns': 1, 'cost': 1, 'gras': 1, 'disp': 1, 'clea': 1, 'fros': 1, 'caro': 1, 'pudd': 1, 'silk': 1, 'prim': 1, 'shir': 1, 'long': 1, 'stre': 1, 'visi': 1, 'hole': 1, 'asia': 1, 'heav': 1, 'shal': 1, 'palu': 1, 'spid': 1, 'stag': 1, 'movi': 1, 'info': 1, 'loca': 1, 'hand': 1, 'ridd': 1, 'scre': 1, 'mati': 1, 'span': 1, 'shri': 1, 'beha': 1, 'pint': 1, 'sink': 1, 'surf': 1, 'atop': 1, 'digg': 1, 'spar': 1, 'isla': 1, 'spor': 1, 'norw': 1, 'west': 1, 'bird': 1, 'stum': 1, 'swal': 1, 'fron': 1, 'nigh': 1, 'book': 1, 'cali': 1, 'pair': 1, 'seas': 1, 'wint': 1, 'pain': 1, 'drin': 1, 'spru': 1, 'fini': 1, 'reme': 1, 'coas': 1, 'land': 1, 'smal': 1, 'prop': 1, 'blen': 1, 'vide': 1, 'piec': 1, 'drag': 1, 'keep': 1, 'rare': 1, 'rica': 1, 'call': 1, 'nati': 1, 'wolf': 1, 'grea': 1, 'mcph': 1, 'sket': 1, 'quit': 1, 'nebu': 1, 'care': 1, 'card': 1, 'take': 1, 'brow': 1, 'webs': 1, 'prin': 1, 'sunn': 1, 'krug': 1, 'pygm': 1, 'nect': 1, 'fogg': 1, 'fire': 1, 'spir': 1, 'mexi': 1, 'migh': 1, 'acto': 1, 'hams': 1, 'logo': 1, 'lang': 1, 'ligh': 1, 'leis': 1, 'zone': 1, 'fear': 1, 'cuba': 1, 'memb': 1, 'fenn': 1, 'tric': 1, 'tire': 1, 'cord': 1, 'mirr': 1, 'thou': 1, 'blue': 1, 'side': 1, 'twig': 1, 'port': 1, 'kids': 1, 'rang': 1, 'illu': 1, 'near': 1, 'comp': 1, 'goph': 1, 'lunc': 1, 'mari': 1, 'shoo': 1, 'carn': 1, 'blac': 1, 'moth': 1, 'fres': 1, 'thai': 1, 'mani': 1, 'chin': 1, 'touc': 1, 'doll': 1, 'defe': 1, 'amar': 1, 'dese': 1, 'refl': 1, 'dart': 1, 'fill': 1, 'hill': 1, 'wear': 1, 'tall': 1, 'behi': 1, 'imag': 1, 'wild': 1, 'sout': 1, 'vulg': 1, 'givi': 1, 'weep': 1, 'delt': 1, 'rail': 1, 'pure': 1, 'stay': 1, 'roya': 1, 'octo': 1, 'chan': 1, 'lila': 1, 'fore': 1, 'born': 1, 'atla': 1, 'rema': 1, 'hall': 1, 'popu': 1, 'ange': 1, 'tear': 1, 'slow': 1, 'made': 1, 'patt': 1, 'agil': 1, 'list': 1, 'edga': 1, 'guit': 1, 'teac': 1, 'sens': 1, 'rege': 1, 'adam': 1, 'comb': 1, 'exam': 1, 'soli': 1, 'judg': 1, 'pter': 1, 'spon': 1, 'righ': 1, 'carr': 1, 'foll': 1, 'frie': 1, 'patc': 1, 'terr': 1, 'habi': 1, 'clou': 1, 'outs': 1, 'wetl': 1, 'morn': 1, 'litt': 1, 'pics': 1, 'forw': 1, 'bute': 1, 'lone': 1, 'whee': 1, 'feli': 1, 'heig': 1, 'rich': 1, 'cour': 1, 'eyes': 1, 'herm': 1, 'earl': 1, 'midd': 1, 'acac': 1, 'char': 1, 'fash': 1, 'melt': 1, 'succ': 1, 'jean': 1, 'caer': 1, 'exce': 1, 'cond': 1, 'unit': 1, 'dapp': 1, 'gate': 1, 'redd': 1, 'coin': 1, 'flor': 1, 'like': 1, 'hair': 1, 'rain': 1, 'camo': 1, 'hang': 1, 'gian': 1, 'mush': 1, 'pred': 1, 'prep': 1, 'conc': 1, 'show': 1, 'craw': 1, 'palm': 1, 'froz': 1, 'cont': 1, 'davi': 1, 'york': 1, 'trea': 1, 'over': 1, 'want': 1, 'paci': 1, 'jaws': 1, 'alps': 1, 'beli': 1, 'tida': 1, 'coyo': 1, 'perf': 1, 'larg': 1, 'minu': 1, 'quee': 1, 'bare': 1, 'spri': 1, 'maki': 1, 'trav': 1, 'give': 1, 'moor': 1, 'sunr': 1, 'even': 1, 'geog': 1, 'digi': 1, 'slot': 1, 'hosp': 1, 'usua': 1, 'pati': 1, 'texa': 1, 'dutc': 1, 'ahea': 1, 'lank': 1, 'sick': 1, 'bisc': 1, 'filt': 1, 'winn': 1, 'infr': 1, 'laye': 1, 'high': 1, 'food': 1, 'eats': 1, 'fort': 1, 'peac': 1, 'fest': 1, 'stoc': 1, 'germ': 1, 'hold': 1, 'east': 1, 'wide': 1, 'colo': 1, 'floo': 1, 'trai': 1, 'dirt': 1, 'duri': 1, 'bamb': 1, 'wren': 1, 'actu': 1, 'beco': 1, 'appe': 1, 'proj': 1, 'mada': 1, 'toge': 1, 'lemu': 1, 'cage': 1, 'besi': 1, 'jock': 1, 'biol': 1, 'tama': 1, 'poli': 1, 'anth': 1, 'ripe': 1, 'moon': 1, 'goin': 1, 'pass': 1, 'firs': 1, 'come': 1, 'peop': 1, 'corn': 1, 'back': 1, 'tast': 1, 'clim': 1, 'woma': 1, 'uniq': 1, 'typi': 1, 'prot': 1, 'sulp': 1, 'tide': 1, 'roug': 1, 'boar': 1, 'pack': 1, 'zeal': 1, 'find': 1, 'farm': 1, 'vert': 1, 'cool': 1, 'floa': 1, 'some': 1, 'sold': 1, 'shop': 1, 'sala': 1, 'pill': 1, 'pave': 1, 'shad': 1, 'vall': 1, 'hori': 1, 'puff': 1, 'name': 1, 'bene': 1, 'colu': 1, 'reci': 1, 'ariz': 1, 'attr': 1, 'wagt': 1, 'soup': 1, 'scel': 1, 'culi': 1, 'sagu': 1, 'spik': 1, 'worl': 1, 'phot': 1, 'favo': 1, 'mode': 1, 'pers': 1, 'edge': 1, 'cany': 1, 'boun': 1, 'rele': 1, 'will': 1, 'ches': 1, 'plas': 1, 'flam': 1, 'mang': 1, 'pest': 1, 'inse': 1, 'plac': 1, 'marc': 1, 'link': 1, 'lyin': 1, 'mont': 1, 'hour': 1, 'cell': 1, 'read': 1, 'publ': 1, 'ceda': 1, 'dune': 1, 'crea': 1, 'stud': 1, 'done': 1, 'uses': 1, 'wire': 1, 'deta': 1, 'stel': 1, 'mald': 1, 'soak': 1, 'auth': 1, 'core': 1, 'kook': 1, 'amid': 1, 'sequ': 1, 'gree': 1, 'ofte': 1, 'alon': 1, 'comm': 1, 'tatt': 1, 'town': 1, 'clif': 1, 'dece': 1, 'upsi': 1, 'famo': 1, 'stop': 1, 'with': 1, 'pric': 1, 'vict': 1, 'harv': 1, 'ingr': 1, 'squa': 1, 'chri': 1, 'seem': 1, 'turn': 1, 'boug': 1, 'hint': 1, 'love': 1, 'also': 1, 'gets': 1, 'onta': 1, 'henn': 1, 'enou': 1, 'went': 1, 'lost': 1, 'plen': 1, 'cobr': 1, 'ojay': 1, 'orio': 1, 'newe': 1, 'cuck': 1, 'sydn': 1, 'hott': 1, 'anhi': 1, 'nice': 1, 'recu': 1, 'pile': 1, 'flat': 1, 'size': 1, 'sing': 1, 'lett': 1, 'cine': 1, 'hard': 1, 'reds': 1, 'haze': 1, 'turk': 1, 'team': 1, 'leat': 1, 'fall': 1, 'fast': 1, 'trum': 1, 'thin': 1, 'coun': 1, 'virg': 1, 'toma': 1, 'huge': 1, 'much': 1, 'held': 1, 'majo': 1, 'most': 1, 'kale': 1, 'wood': 1, 'girl': 1, 'them': 1, 'noth': 1, 'resi': 1, 'effe': 1, 'droo': 1, 'rega': 1, 'arou': 1, 'site': 1, 'birt': 1, 'step': 1, 'beet': 1, 'past': 1, 'camp': 1, 'wedd': 1, 'whit': 1, 'amaz': 1, 'down': 1, 'home': 1, 'elec': 1, 'expe': 1, 'stro': 1, 'easi': 1, 'powe': 1, 'gent': 1, 'deco': 1, 'arid': 1, 'rays': 1, 'seam': 1, 'espe': 1, 'fant': 1, 'craz': 1, 'hate': 1, 'ugan': 1, 'phon': 1, 'owls': 1, 'lemo': 1, 'cera': 1, 'sunl': 1, 'engr': 1, 'teet': 1, 'mack': 1, 'brok': 1, 'next': 1, 'film': 1, 'know': 1, 'gift': 1, 'shou': 1, 'desi': 1, 'devo': 1, 'eart': 1, 'mand': 1, 'brit': 1, 'cong': 1, 'beas': 1, 'dess': 1, 'stil': 1, 'livi': 1, 'late': 1, 'stun': 1, 'toda': 1, 'cave': 1, 'inva': 1, 'brid': 1, 'bell': 1, 'mark': 1, 'ever': 1, 'bore': 1, 'deep': 1, 'cove': 1, 'amon': 1, 'cani': 1, 'lupu': 1, 'mist': 1, 'simi': 1, 'summ': 1, 'brac': 1, 'lodg': 1, 'vipe': 1, 'raja': 1, 'vehi': 1, 'hopp': 1, 'popl': 1, 'dust': 1, 'miss': 1, 'pois': 1, 'slop': 1, 'adap': 1, 'clar': 1, 'incl': 1, 'bung': 1, 'main': 1, 'natu': 1, 'holi': 1, 'save': 1, 'scot': 1, 'anot': 1, 'rott': 1, 'emer': 1, 'bark': 1, 'geck': 1, 'cano': 1, 'area': 1, 'spla': 1, 'reac': 1, 'clos': 1, 'russ': 1, 'mile': 1, 'arti': 1, 'fran': 1, 'fiel': 1, 'pere': 1, 'plat': 1, 'outl': 1, 'gold': 1, 'make': 1}
plant_list = {'vase': 1, 'bego': 1, 'papa': 1, 'gera': 1, 'rosa': 1, 'arro': 1, 'frin': 1, 'ixor': 1, 'soyb': 1, 'clem': 1, 'forg': 1, 'raid': 1, 'borg': 1, 'minc': 1, 'vene': 1, 'sutt': 1, 'flak': 1, 'mano': 1, 'quak': 1, 'mora': 1, 'glea': 1, 'cele': 1, 'bien': 1, 'ilex': 1, 'arun': 1, 'bole': 1, 'simu': 1, 'prie': 1, 'grow': 1, 'plan': 1, 'bord': 1, 'blow': 1, 'avoc': 1, 'rose': 1, 'bloo': 1, 'wist': 1, 'twin': 1, 'deli': 1, 'undu': 1, 'mess': 1, 'flas': 1, 'mort': 1, 'peon': 1, 'hede': 1, 'azal': 1, 'jewe': 1, 'barl': 1, 'blos': 1, 'pine': 1, 'clum': 1, 'more': 1, 'lanc': 1, 'oxfo': 1, 'flow': 1, 'leav': 1, 'lotu': 1, 'frag': 1, 'glos': 1, 'gors': 1, 'sewn': 1, 'cauc': 1, 'rhod': 1, 'jaca': 1, 'stra': 1, 'heli': 1, 'vein': 1, 'toxi': 1, 'eryn': 1, 'jade': 1, 'peta': 1, 'sunf': 1, 'crop': 1, 'deca': 1, 'perg': 1, 'pinu': 1, 'arum': 1, 'weed': 1, 'worc': 1, 'arom': 1, 'drap': 1, 'orie': 1, 'maju': 1, 'fung': 1, 'frui': 1, 'berr': 1, 'ivy': 1, 'moss': 1, 'maiz': 1, 'oak': 1, 'lily': 1, 'pars': 1, 'bell': 1, 'pois': 1, 'gard': 1, 'elm': 1, 'birc': 1, 'clov': 1, 'cara': 1, 'cott': 1, 'drie': 1, 'cut': 1, 'hibi': 1, 'leaf': 1}
food_list = {'papa': 1, 'maiz': 1, 'cele': 1, 'chiv': 1, 'broc': 1, 'stra': 1, 'ging': 1, 'avoc': 1, 'pars': 1, 'basi': 1, 'hibi': 1, 'rose': 1, 'almo': 1, 'oliv': 1, 'roas': 1, 'onio': 1, 'cara': 1, 'peac': 1, 'cook': 1, 'quin': 1, 'jack': 1, 'bowl': 1, 'deli': 1, 'herb': 1, 'slic': 1, 'reci': 1, 'flav': 1, 'oven': 1, 'dish': 1, 'pot': 1, 'tast': 1, 'boil': 1, 'garl': 1, 'plat': 1}

replace_lst = [("internet publishing and broadcasting and web search portals business", "internet"), \
           ("automotive industry business", "car company"), \
           ("advertising agencies business", "advertising agency"), \
           ("medicinal and botanical manufacturing business", "a company"), \
           ("manufacturing business", "a company"), \
           ("bodies of water", "waters"),
           ("automobile model", "car"),
           ("a body of water", "the waters"),
           ("body of water", "waters"),
           ]
#TODO
#fix:
# '-- stock #', 
# 'show as part', 
# 'sports association', 
# 'automotive industry business', 

keywordLstHash ={}
keywordLst2Hash = {}
all_words = []
ontology_inv={'accessory': ['boot',
  'sneakers',
  'glove',
  'suitcase',
  'heels',
  'tie',
  'accessory',
  'clutch',
  'shoe',
  'tote',
  'bag',
  'umbrella',
  'sombrero',
  'purse',
  'glasses',
  'converse',
  'backpack',
  'handbag',
  'flippers',
  'briefcase'],
 'animal': ['unicorn','ostrich',
  'bear',
  'python',
  'lamb',
  'coyote',
  'wildebeest',
  'horse',
  'zebra',
  'reindeer',
  'grizzly',
  'alligator', 'crocodile',
  'lemur',
  'sheep',
  'hyena',
  'insect','butterfly','dragonfly',
  'animal', 'lion', 'tiger', 'leopard', 'lynx',
  'tabby',
  'prey',
  'predator',
  'bumblebee',
  'giraffe',
  'hippos',
  'jaguar',
  'gorilla',
  'macaw',
  'cubs',
  'koala',
  'flea',
  'elephant',
  'beetle',
  'cheetah',
  'cow',
  'spider'],
 'bird': ['swan',
  'seabird',
  'osprey',
  'heron',
  'chicken',
  'rooster',
  'kingfisher',
  'duckling',
  'seagull',
  'duck','goose','geese',
  'toucan',
  'bird',
  'coral',
  'hen'],
 'building': ['building','chapel',
  'architecture',
  'windmill',
  'apartment',
  'synagogue',
  'hut',
  'house',
  'treehouse',
  'manger',
  'farmhouse',
  'livery',
  'cabin',
  'pagoda',
  'watchtower',
  'church',
  'girder',
  'manor',
  'mansion',
  'aquarium',
  'home',
  'mosque',
  'school'],
 'business': ['museum','airline','hospital',
  'retailer',
  'retail',
  'healthcare',
  'venture',
  'motel',
  'aerospace',
  'drugstore',
  'theater',
  'store',
  'manufacturer',
  'networks',
  'pharmacy',
  'diner',
  'partnership',
  'company',
  'hotel',
  'restaurant', 'pub',
  'production',
  'business',
  'engineering',
  'barbershop','shop',
  'industry'],
 'clothes': ['jacket','jeans'
  'raincoat',
  'dress',
  'menswear',
  'swimwear',
  'bodice',
  'clothes',
  'fashion',
  'shirt',
  'tunic',
  'laundry',
  'pants',
  'halter',
  'sarong',
  'wetsuit',
  'sweater',
  'coat',
  'hijab'],
 'appliance':[
  'microwave',
  'oven',
  'refrigerator',
  'appliance',
 ],
 'device': ['cellphone',
  'hardware',
  'electronics',
  'blower',
  'sprinkler',
  'transmission',
  'computer',
  'bumper',
  'clock',
  'pylon',
  'mri',
  'turbine',
  'alarm',
  'phone',
  'dashboard',
  'handcuff',
  'tablet',
  'motherboard',
  'speedometer',
  'laptop',
  'engine',
  'propeller',
  'processor',
  'toaster',
  'washer',
  'desktop',
  'device'],
 'drink': ['brandy',
  'martini',
  'whiskey',
  'latte',
  'coffee',
  'cola',
  'drink',
  'espresso','cappuccino',
  'nectar',
  'drinks',
  'soda',
  'alcohol',
  'milk'],
 'event': ['film','weather','protest',
  'interview',
  'cup',
  'spacewalk',
  'concert',
  'cinema',
  'wedding',
  'combat',
  'awards',
  'ceremony',
  'meal',
  'rain',
  'gridlock',
  'premiere',
  'game',
  'dedication',
  'resignation',
  'session',
  'deployment',
  'enforcement',
  'audition',
  'campaign',
  'olympic',
  'quest',
  'inning',
  'reception',
  'heptathlon',
  'sxsw',
  'dinner',
  'election',
  'art',
  'rainstorm',
  'recreation',
  'airshow',
  'traffic',
  'sale',
  'contest',
  'match',
  'transit',
  'heartland',
  'slalom',
  'riot',
  'event',
  'show',
  'renaissance',
  'picnic',
  'speech',
  'playoff',
  'birthday',
  'fifa',
  'episode',
  'arrival',
  'festival',
  'christmas',
  'television',
  'accident',
  'graduation',
  'opera',
  'charity',
  'afterparty','party',
  'fis',
  'commencement',
  'investigation',
  'sunset',
  'initiative',
  'debate',
  'broadway'],
 'food': ['cheddar', 'pasta','sausage',
  'cheese',
  'muffin',
  'salad',
  'feta',
  'burger',
  'roast',
  'fondant',
  'sandwich',
  'buttercream',
  'ketchup',
  'curry',
  'meat',
  'mozzarella',
  'pizza',
  'waffle',
  'bread',
  'pancake','cookie',
  'dessert',
  'donut',
  'rice',
  'cheeseburger',
  'pepperoni',
  'gingerbread',
  'soup',
  'toppings',
  'carb',
  'grocery',
  'dumpling',
  'food','cuisine','chocolate','candy',
  'buffet',
  'cake',
  'snack',
  'steak',
  'pastry',
  'loaf',
  'meatball',
  'topper',
  'tobacco',
  'noodle'],
 'fruit': ['oranges',
  'plums',
  'kiwi',
  'apple', 'pineapple',
  'fruit',
  'orange',
  'coconut',
  'soybean',
  'acorn',
  'watermelon',
  'peanut',
  'figs',
  'banana',
  'avocado',
  'olive',
  'pear',
  'strawberry',
  'almond',
  'citrus','lemon','lime',
  'chestnut',
  'cherry',
  'mango',
  'apricot'],
 'furniture': ['headboard',
  'sink',
  'table',
  'divider',
  'furniture','sofa',
  'armchair',
  'jacuzzi',
  'bed',
  'mantle',
  'chair',
  'couch',
  'bunk',
  'cupboard',
  'console',
  'bathtub',
  'toilet',
  'windscreen',
  'bench'],
 'group': ['bunch', 'trio', 'cluster',
  'group',
  'convoy',
  'flock',
  'set',
  'couple',
  'collection',
  'crowd',
  'herd'],
 'instrument': ['keyboard',
  'guitar',
  'drum',
  'instrument',
  'trumpet',
  'saxophone'],
 'landscape': ['savanna','countryside',
  'sky',
  'meadow',
  'redwood',
  'summit',
  'ridge',
  'dirt',
  'park',
  'garden',
  'snow',
  'lava',
  'volcano',
  'outdoor',
  'paradise',
  'badland',
  'borealis',
  'iceberg',
  'landscape',
  'sand',
  'snowfall',
  'skyline',
  'mountain',  'mountainside', 'range',
  'grass',
  'farm',
  'heartland',
  'constellation',
  'island',
  'forest',
  'farmyard',
  'woodland','marshland','wetland',
  'geyser',
  'valley',
  'hill',
  'treetop',
  'field',
  'peak'],
 'location': ['tarmac','villa',
  'market',
  'vicinity',
  'airport',
  'dealership',
  'seawall',
  'airfield',
  'pool',
  'pothole',
  'shipyard',
  'bedside',
  'port',
  'wall',
  'stage',
  'crater',
  'gate',
  'region',
  'spillway',
  'tomb',
  'exit','entrance',
  'haystack',
  'attraction',
  'floor',
  'churchyard',
  'court',
  'bridge','footbridge',
  'location','mall',
  'mural',
  'area', 'earth',
  'marketplace',
  'sydney'],
 'music': ['rap', 'folk', 'electronica', 'protopunk', 'music', 'jazz', 'rock'],
 'path': ['path',
  'asphalt',
  'gravel',
  'motorway',
  'footpath',
  'runway',
  'racetrack',
  'road',
  'track',
  'cobblestone',
  'railroad','railway',
  'street'],
 'people': ['women', 'men', 'people', 'children', 'audience', 'family',],
 'person': ['stranger', 'son', 'daughter', 'rider','caucasian','customer','supporter',
  'child',
  'messenger',
  'mother',
  'patient',
  'bridesmaid',
  'tourist',
  'communist',
  'student',
  'biker',
  'buyer',
  'skier',
  'kid',
  'reveler',
  'woman',
  'buddha',
  'teenager',
  'adult',
  'enthusiasts',
  'character',
  'friend','girlfriend','boyfriend',
  'person',
  'hikers',
  'girl',
  'fans',
  'man',
  'gangsta',
  'fool',
  'baby',
  'motorcyclist',
  'protester',
  'boy',
  'dealer',
  'demonstrator',
  'parent',
  'jogger',
  'guardian','guest',
  'children', 'son', 'daugther','sister','brother',
  'surfer',
  'father','grandmother','grandfather',
  'bride',
  'newborn',
  'passenger',
  'groom',
  'christian',
  'costumer',
  'catholic',
  'observer'],
 'pet': ['pug',
  'chihuahua',
  'labrador',
  'dog',
  'husky',
  'collie',
  'kitten',
  'pet',
  'kitty',
  'cat',
  'spaniel',
  'poodle',
  'terrier',
  'beagle',
  'cocker',
  'calico'],
 'plant': ['bloom',
  'lily',
  'chrysanthemum',
  'lotus',
  'deciduous',
  'orchid',
  'evergreen',
  'eucalyptus',
  'bonsai',
  'buttercup',
  'fungi',
  'beech',
  'carnation',
  'elm',
  'fungus',
  'dahlia',
  'acacia',
  'cottonwood',
  'magnolia',
  'blossom',
  'tree',
  'firs',
  'birch',
  'banyan',
  'lavender',
  'plant',
  'dandelion',
  'aloe',
  'peony',
  'hibiscus',
  'lichen',
  'baobab',
  'flower',
  'bamboo',
  'bud',
  'cypress',
  'cactus',
  'cedar',
  'oak',
  'sunflower', 'wildflower',
  'saguaro'],
 'military': ['troop', 'battalion' , 'military', 'regiment', 'generals',],
 'professional': ['professional', 'athlete', 'boxer', 'model','referee','florist','engineer','workman','workwoman','workmen','workwomen','author','celebrities','celebrity',
  'songwriter','commander','soldier', 
  'linebacker',
  'waitress',
  'paraglider',
  'athlete',
  'chauffeur',
  'king',
  'bassist',
  'spokesperson',
  'worker',
  'nurse',
  'clown',
  'actor',
  'farmer',
  'firemen',
  'equestrian',
  'butcher',
  'cheerleader','leader',
  'lifeguard',
  'goaltender',
  'impresario',
  'frontman',
  'publicist',
  'queen',
  'astronomer',
  'beekeeper',
  'preacher',
  'salesman',
  'programmer',
  'politician', 'president', 'minister',
  'physician',
  'buddhist',
  'climber',
  'comedian',
  'skater',
  'scriptwriter',
  'samurai',
  'emperor',
  'pilot',
  'cinematographer',
  'drummer',
  'guitarist',
  'businessman',
  'mahout',
  'receiver',
  'screenwriter',
  'hairdresser',
  'pharmacist',
  'waiter',
  'businesswoman',
  'vocalist',
  'promoter',
  'fireman',
  'defenceman',
  'sitter',
  'lecturer',
  'director',
  'shortstop',
  'doctor',
  'artist',
  'jockey',
  'producer',
  'sculptor',
  'shepherd',
  'engineer',
  'driver',
  'penciler',
  'police',
  'mechanic',
  'violinist',
  'cricketer',
  'developer',
  'chef','cook',
  'journalist',
  'faculty',
  'surgeon',
  'paramedic',
  'actress',
  'painter',
  'golfer',
  'operator',
  'astronaut',
  'officer',
  'player','footballer',
  'mountaineer'],
 'room': ['bedroom',
  'lab',
  'loft',
  'cockpit',
  'closet',
  'pulpit',
  'office',
  'room',
  'ensuite',
  'kennel',
  'bathroom',
  'altar',
  'locker',
  'dorm',
  'kitchen'],
 'fish': ['koi','jellyfish',
  'conch',
  'eel',
  'anemone',
  'fish', 'salmon','crab','octopus','squid',
  'shrimp',
  'seal',
  'dolphin'],
 'season': ['season', 'autumn', 'winter', 'holiday', 'summer', 'spring'],
 'sport': ['basketball',
  'karate',
  'golf',
  'sport',
  'baseball',
  'soccer',
  'handball',
  'football',
  'darts',
  'billiard'],
 'team': ['team'],
 'kitchenware': ['cork','kitchenware',
  'pan',
  'wok',
  'skillet',
  'spatula',
  'knife',
  'vase',
  'pot',
  'corkscrew',
  'bowl',
  'bottle',
  'fork',
  'mug',
  'spoon',
  'dish',],
 'tool': ['cork',  'tool',
  'perfume',
  'screwdriver',
  'paintbrush',
  'toothpick',
  'scissors',
  'toothbrush',
  'dryer',
  'easel',
  'crayon',
  'zipper',
  'marker',
  'mousetrap',
  'comic',
  'brush',
  'compass',
  'tweezer',
  'toothpaste'],
 'town': ['village', 'london', 'town', 'city', 'municipality', 'capitol',],
 'toy': ['toy',
  'surfboard',
  'chess',
  'pawn',
  'kite',
  'frisbee',
  'puck',
  'racket',
  'bat',
  'poker',
  'skis',
  'teddy',
  'chessboard',
  'puppet',
  'snowboard',
  'ball',
  'paddle',
  'skateboard',
  'skate',
  'dartboard'],
 'vegetable': ['broccoli','zucchini',
  'tomato',
  'barley',
  'carrot',
  'vegetable',
  'corn',
  'cauliflower',
  'wheat',
  'herbs',
  'canola',
  'radish',
  'celery',
  'pumpkin'],
 'vehicle': ['bicycle',
  'mower',
  'bus',
  'ambulance',
  'rickshaw',
  'yacht',
  'snowmobile',
  'corvette',
  'plane',
  'vessel',
  'ferry',
  'tug',
  'sedan',
  'boat',
  'pontoon',
  'train',
  'locomotive',
  'sleigh',
  'tractor',
  'ship',
  'motorcycle',
  'airplane',
  'convertible',
  'sidecar',
  'sailboat','watercraft',
  'vehicle',
  'frigate',
  'carrier',
  'gondola',
  'suv',
  'chairlift',
  'car',
  'lawnmower',
  'automobile',
  'carriage',
  'supercar',
  'truck','pickup',
  'aircraft',
  'jet'],
 'view': ['view', 'scene'],
 'waters': ['flood', 'wave',
  'atlantic',
  'backwater',
  'pacific',
  'coast',
  'waters',
  'reef',
  'river','stream',
  'ocean',
  'breakwater',
  'freshwater',
  'sea',
  'beach',
  'seascape',
  'bay',
  'gulf',
  'mediterranean',
  'oceanfront',
  'mangrove',
  'lake']}

ontology = {}
for item in ontology_inv.items():
  for i in item[1]:
      if i not in ontology:
          ontology[i] = item[0]

ignore_category = ('person', "background", "event")
keywords = list(set(list(ontology.keys()) + list(ontology.values())))
all_words_caption=[]
          
ignore_generic = {'shop':1, 'model': 1, 'boot': 6775, 'tie': 4624, 'shoe': 11255, 'bag': 8521, 'umbrella': 6135, 'glasses': 12080, 'bear': 9563, 'horse': 27248, 'sheep': 5710, 'animal': 18682, 'elephant': 4444, 'coral': 4643, 'bird': 10231, 'chicken': 4196, 'building': 48817, 'church': 12590, 'architecture': 5311, 'apartment': 11305, 'house': 67125, 'home': 66228, 'school': 18993, 'museum': 5309, 'theater': 4353, 'store': 8710, 'company': 10571, 'hotel': 11148, 'pub':1, 'restaurant': 13867, 'business': 32634, 'industry': 11599, 'jeans':1, 'jacket': 8121, 'dress': 41002, 'clothes': 5549, 'fashion': 46099, 'shirt': 13904, 'coat': 7010, 'computer': 13494, 'clock': 6347, 'phone': 15321, 'tablet': 8613, 'laptop': 14314, 'coffee': 16880, 'drink': 6109, 'film': 57214, 'cup': 12501, 'concert': 27371, 'wedding': 41150, 'awards': 25032, 'ceremony': 17104, 'meal': 4456, 'rain': 16226, 'premiere': 83288, 'game': 81012, 'session': 18558, 'olympic': 11113, 'inning': 6241, 'dinner': 7564, 'art': 31256, 'traffic': 11761, 'sale': 16214, 'match': 29684, 'event': 46885, 'show': 55194, 'birthday': 15282, 'festival': 51459, 'christmas': 35224, 'graduation': 5771, 'sunset': 48819, 'food': 26953, 'cake': 17656, 'apple': 4775, 'fruit': 8239, 'orange': 13431, 'table': 41835, 'furniture': 5757, 'bed': 18380, 'chair': 8850, 'couch': 5259, 'bench': 10342, 'group': 7656, 'set': 18524, 'couple': 39132, 'collection': 11001, 'crowd': 16219, 'guitar': 28340, 'instrument': 60562, 'sky': 61449, 'meadow': 10083, 'park': 42386, 'garden': 36386, 'snow': 43413, 'outdoor': 16411, 'landscape': 20416, 'sand': 15478, 'skyline': 10442, 'range': 1, 'mountain': 32355, 'grass': 28723, 'farm': 12065, 'island': 16351, 'forest': 42024, 'valley': 6724, 'hill': 8500, 'field': 46015, 'market': 19051, 'airport': 5577, 'pool': 17321, 'port': 5361, 'wall': 35964, 'stage': 95163, 'attraction': 19823, 'floor': 24675, 'court': 4158, 'bridge': 27357, 'location': 14739, 'area': 22519, 'region': 6107, 'folk': 4259, 'music': 17396, 'rock': 44930, 'path': 9128, 'street': 45293, 'road': 50718, 'track': 4235, 'runway': 18750, 'women': 19901, 'men': 17136, 'people': 82041, 'child': 20874, 'mother': 18009, 'student': 5683, 'kid': 13103, 'woman': 115071, 'friend': 8169, 'cook':1, 'person': 458123, 'family':1, 'girl': 84053, 'fans': 17915, 'man': 129424, 'baby': 24878, 'boy': 38889, 'teenage': 4847, 'parent': 5104, 'children': 23342, 'grandmother':1, 'grandfather':1, 'son':1, 'daugther':1,'sister':1,'brother':1,'father': 10346, 'bride': 19409, 'groom': 10498, 'christian': 8081, 'tourist': 23160, 'character': 25976, 'dog': 41488, 'kitten': 4611, 'cat': 25756, 'tree': 82933, 'plant': 12184, 'flower': 22656, 'professional': 5475, 'athlete': 23377, 'worker': 4444, 'actor': 141374, 'politician': 27879, 'comedian': 8982, 'businessman': 12620, 'director': 9355, 'artist': 135980, 'driver': 4513, 'police': 14547, 'golfer': 4523, 'player': 122234, 'bedroom': 18660, 'office': 20165, 'room': 67947, 'bathroom': 12290, 'kitchen': 28307, 'fish': 14684, 'season': 19620, 'autumn': 23284, 'winter': 35119, 'holiday': 21952, 'summer': 35156, 'spring': 20493, 'basketball': 26323, 'golf': 5563, 'sport': 19697, 'baseball': 22347, 'soccer': 27749, 'football': 106971, 'team': 74323, 'bowl': 7905, 'bottle': 9386, 'comic': 5640, 'dish': 8323, 'village': 15011, 'town': 23003, 'city': 107074, 'ball': 45517, 'vegetable': 8168, 'wheat': 6363, 'bicycle': 7384, 'bus': 8007, 'aircraft': 8321, 'plane': 4685, 'boat': 27574, 'train': 16965, 'tractor': 4158, 'ship': 15891, 'motorcycle': 5362, 'airplane': 6125, 'vehicle': 7546, 'car': 75351, 'truck': 9165, 'view': 35662, 'scene': 10272, 'coast': 16454, 'waters': 8473, 'reef': 4232, 'river': 42846, 'ocean': 15737, 'sea': 47319, 'beach': 89054, 'bay': 5510, 'stream':1, 'lake': 29069, 'earth':1, }

from nltk.corpus import stopwords
stopwords_hash=set(stopwords.words('english'))

def recreate_key_words():
  keywords = [a["name"] for a in json.load(open("/content/drive/MyDrive/knwoledge_transformers/cocoontology.json")).values()]
  keywords = list(set(keywords + ['meat', 'fish', 'steak', 'toy', 'motorcycle', 'fruit', 'plane', 'bus', 'snack', 'vegetable', 'skateboard', 'food',  'ball', 'film',  \
                      'tv', 'car',  'plant', 'train', 'handbag', 'bag', 'toy', 'light', 'laptop',  'computer', 'cup', 'knife', 'couch', 'sink', 'tool', \
                      'microwave', 'vase', 'oven', 'device', 'bird', 'book', 'bed', 'bottle', 'umbrella', 'backpack',  'boat', 'frisbee', 'oven', 'table',\
                      'horse', 'team', 'meter', 'suitcase', 'clock',  'company', 'insect', 'cup', 'racket', 'bench', 'airplane', 'keyboard', 'dryer', 'mouse',\
                      'hotdog', 'toilet', 'surfboard', 'truck', 'business', 'refrigerator',  'toaster', 'hydrant', 'bicycle', 'chair', 'city', 'animal', ]))
  print (keywords)


def recreate_ontology_from_text():
  vocab = Counter(all_words_caption)
  words = list(vocab.items())
  words.sort(key=lambda a:a[1], reverse=True)
  top25per = words[:int(len(words)*.25)]
  top25per = dict(top25per)
  ontology = {}
  for item in ontology_inv.items():
    for i in item[1]:
      ontology[i] = item[0]
  all_words = []
  keywordLst={}
  #a in ignore_cap  or 
  for key in sorted(list(keywordLstHash.keys())):
    new_words = [a for a in keywordLstHash[key] if not a in stopwords_hash and len(a) > 2]
    new_words = [a for a in new_words if (a in top25per or a+"s" in top25per or a+"es" in top25per) and not (a in ontology or a in ontology_inv or a+"s" in ontology or a+"s" in ontology_inv or a+"es" in ontology or a+"es" in ontology_inv or a.endswith("ous") or a.endswith("al") or a.endswith("ed") or a.endswith("ing") or a.endswith("es") or a.endswith("less") or a.endswith("ful"))]
    all_words.extend(new_words)
    keywordLst[key] = Counter(new_words)
  all_words = Counter(all_words)

  for item in keywordLst.items():
    aHash = item[1]
    for key in list(aHash.keys()):
      if aHash[key] < 4:
        del aHash[key]
      else:
        aHash[key] /= all_words[key]
        if aHash[key] <0.6:
            del aHash[key] 
    print (item)



def create_ontology_from_wn():
  professionals = [[str(s1.name()).lower() for s1 in s.lemmas()] for s in wn.synset('professional.n.01').closure(lambda s: s.hyponyms())]
  professionals = list(itertools.chain(*professionals)) 
  professionals_hash = dict([(a,1) for a in professionals])
  bird = [[str(s1.name()).lower() for s1 in s.lemmas()] for s in wn.synset('bird.n.01').closure(lambda s: s.hyponyms())]
  bird = list(itertools.chain(*bird))
  bird_hash = dict([(a,1) for a in bird])
  insect = [[str(s1.name()).lower() for s1 in s.lemmas()] for s in wn.synset('insect.n.01').closure(lambda s: s.hyponyms())] + \
        [[str(s1.name()).lower() for s1 in s.lemmas()] for s in wn.synset('arachnid.n.01').closure(lambda s: s.hyponyms())] 
  insect = list(itertools.chain(*insect))
  insect_hash = dict([(a,1) for a in insect])
  reptile = [[str(s1.name()).lower() for s1 in s.lemmas()] for s in wn.synset('reptile.n.01').closure(lambda s: s.hyponyms())] + \
          [[str(s1.name()).lower() for s1 in s.lemmas()] for s in wn.synset('amphibian.n.03').closure(lambda s: s.hyponyms())] 
  reptile = list(itertools.chain(*reptile))
  reptile_hash = dict([(a,1) for a in reptile])
  pet = [[str(s1.name()).lower() for s1 in s.lemmas()] for s in  wn.synset('dog.n.01').closure(lambda s: s.hyponyms())] + \
        [[str(s1.name()).lower() for s1 in s.lemmas()] for s in  wn.synset('domestic_cat.n.01').closure(lambda s: s.hyponyms())]+ \
        [[str(s1.name()).lower() for s1 in s.lemmas()] for s in  wn.synset('domestic_animal.n.01').closure(lambda s: s.hyponyms())]
  pet = [a for a in itertools.chain(*pet)]
  pet_hash = dict([(a,1) for a in pet])
  #sea_animal = [[str(s1.name()).lower() for s1 in s.lemmas()] for s in wn.synset('sea_animal.n.01').closure(lambda s: s.hyponyms())]
  #sea_animal = list(itertools.chain(*sea_animal)) 

  animals = [[str(s1.name()).lower() for s1 in s.lemmas()] for s in  wn.synset('animal.n.01').closure(lambda s: s.hyponyms())]
  animals = [a for a in itertools.chain(*animals) if a not in professionals_hash and a not in pet_hash and a not in bird_hash and a not in insect_hash and a not in reptile_hash]
  animals_hash = dict([(a,1) for a in animals])
  food = [[str(s1.name()).lower() for s1 in s.lemmas()] for s in  wn.synset('food.n.02').closure(lambda s: s.hyponyms())]
  food = [a for a in itertools.chain(*food) if  a not in animals_hash and  a not in professionals_hash and  a not in person_hash and  a not in pet_hash and a not in bird_hash and a not in insect_hash and a not in reptile_hash]
  person = [[str(s1.name()).lower() for s1 in s.lemmas()] for s in  wn.synset('person.n.01').closure(lambda s: s.hyponyms())]
  person = [a for a in itertools.chain(*person) if a not in professionals_hash and  a not in animals_hash ]
  person_hash = dict([(a,1) for a in person])

  appliances = [[str(s1.name()).lower() for s1 in s.lemmas()] for s in wn.synset('appliance.n.02').closure(lambda s: s.hyponyms())]
  appliances = list(itertools.chain(*appliances)) 
  tableware = [[str(s1.name()).lower() for s1 in s.lemmas()] for s in wn.synset('tableware.n.01').closure(lambda s: s.hyponyms())]
  tableware = list(itertools.chain(*tableware)) 
  clothes = [[str(s1.name()).lower() for s1 in s.lemmas()] for s in wn.synset('garment.n.01').closure(lambda s: s.hyponyms())]
  clothes = list(itertools.chain(*clothes)) 
  beverage = [[str(s1.name()).lower() for s1 in s.lemmas()] for s in wn.synset('beverage.n.01').closure(lambda s: s.hyponyms())]
  beverage = list(itertools.chain(*beverage)) 
  vehicle = [[str(s1.name()).lower() for s1 in s.lemmas()] for s in wn.synset('vehicle.n.01').closure(lambda s: s.hyponyms())]
  vehicle = list(itertools.chain(*vehicle)) 
  waters = [[str(s1.name()).lower() for s1 in s.lemmas()] for s in wn.synset('body_of_water.n.01').closure(lambda s: s.hyponyms())]
  waters = list(itertools.chain(*waters)) 

keywordLstHash ={}
keywordLst2Hash = {}
all_words = []

ignore_generic = {'boot': 6775, 'tie': 4624, 'shoe': 11255, 'bag': 8521, 'umbrella': 6135, 'glasses': 12080, 'bear': 9563, 'horse': 27248, 'sheep': 5710, 'animal': 18682, 'elephant': 4444, 'coral': 4643, 'bird': 10231, 'chicken': 4196, 'building': 48817, 'church': 12590, 'architecture': 5311, 'apartment': 11305, 'house': 67125, 'home': 66228, 'school': 18993, 'museum': 5309, 'theater': 4353, 'store': 8710, 'company': 10571, 'hotel': 11148, 'restaurant': 13867, 'business': 32634, 'industry': 11599, 'jacket': 8121, 'dress': 41002, 'clothes': 5549, 'fashion': 46099, 'shirt': 13904, 'coat': 7010, 'computer': 13494, 'clock': 6347, 'phone': 15321, 'tablet': 8613, 'laptop': 14314, 'coffee': 16880, 'drink': 6109, 'film': 57214, 'cup': 12501, 'concert': 27371, 'wedding': 41150, 'awards': 25032, 'ceremony': 17104, 'meal': 4456, 'rain': 16226, 'premiere': 83288, 'game': 81012, 'session': 18558, 'olympic': 11113, 'inning': 6241, 'dinner': 7564, 'art': 31256, 'traffic': 11761, 'sale': 16214, 'match': 29684, 'event': 46885, 'show': 55194, 'birthday': 15282, 'festival': 51459, 'christmas': 35224, 'graduation': 5771, 'sunset': 48819, 'food': 26953, 'cake': 17656, 'apple': 4775, 'fruit': 8239, 'orange': 13431, 'table': 41835, 'furniture': 5757, 'bed': 18380, 'chair': 8850, 'couch': 5259, 'bench': 10342, 'group': 7656, 'set': 18524, 'couple': 39132, 'collection': 11001, 'crowd': 16219, 'guitar': 28340, 'instrument': 60562, 'sky': 61449, 'meadow': 10083, 'park': 42386, 'garden': 36386, 'snow': 43413, 'outdoor': 16411, 'landscape': 20416, 'sand': 15478, 'skyline': 10442, 'mountain': 32355, 'grass': 28723, 'farm': 12065, 'island': 16351, 'forest': 42024, 'valley': 6724, 'hill': 8500, 'field': 46015, 'market': 19051, 'airport': 5577, 'pool': 17321, 'port': 5361, 'wall': 35964, 'stage': 95163, 'attraction': 19823, 'floor': 24675, 'court': 4158, 'bridge': 27357, 'location': 14739, 'area': 22519, 'region': 6107, 'folk': 4259, 'music': 17396, 'rock': 44930, 'path': 9128, 'street': 45293, 'road': 50718, 'track': 4235, 'runway': 18750, 'women': 19901, 'men': 17136, 'people': 82041, 'child': 20874, 'mother': 18009, 'student': 5683, 'kid': 13103, 'woman': 115071, 'friend': 8169, 'person': 458123, 'girl': 84053, 'fans': 17915, 'man': 129424, 'baby': 24878, 'boy': 38889, 'teenage': 4847, 'parent': 5104, 'children': 23342, 'father': 10346, 'bride': 19409, 'groom': 10498, 'christian': 8081, 'tourist': 23160, 'character': 25976, 'dog': 41488, 'kitten': 4611, 'cat': 25756, 'tree': 82933, 'plant': 12184, 'flower': 22656, 'professional': 5475, 'athlete': 23377, 'worker': 4444, 'actor': 141374, 'politician': 27879, 'comedian': 8982, 'businessman': 12620, 'director': 9355, 'artist': 135980, 'driver': 4513, 'police': 14547, 'golfer': 4523, 'player': 122234, 'bedroom': 18660, 'office': 20165, 'room': 67947, 'bathroom': 12290, 'kitchen': 28307, 'fish': 14684, 'season': 19620, 'autumn': 23284, 'winter': 35119, 'holiday': 21952, 'summer': 35156, 'spring': 20493, 'basketball': 26323, 'golf': 5563, 'sport': 19697, 'baseball': 22347, 'soccer': 27749, 'football': 106971, 'team': 74323, 'bowl': 7905, 'bottle': 9386, 'comic': 5640, 'dish': 8323, 'village': 15011, 'town': 23003, 'city': 107074, 'ball': 45517, 'vegetable': 8168, 'wheat': 6363, 'bicycle': 7384, 'bus': 8007, 'aircraft': 8321, 'plane': 4685, 'boat': 27574, 'train': 16965, 'tractor': 4158, 'ship': 15891, 'motorcycle': 5362, 'airplane': 6125, 'vehicle': 7546, 'car': 75351, 'truck': 9165, 'view': 35662, 'scene': 10272, 'coast': 16454, 'waters': 8473, 'reef': 4232, 'river': 42846, 'ocean': 15737, 'sea': 47319, 'beach': 89054, 'bay': 5510, 'lake': 29069}
ontology = {}

def test_ontology(ontology_inv, test_lst = ['the front of the house with the wrap - around deck']):
  for item in ontology_inv.items():
    for i in item[1]:
      ontology[i] = item[0]
  ignore_category = ('person', "event")
  keywords = list(set(list(ontology.keys()) + list(ontology.values())))
  for txt2 in (test_lst):
    txt = parse_with_ontology(txt2, ontology, ignore_category, keywords)
    print (txt2, '***', txt)

def parse_with_ontology(txt2, ontology, ignore_category, keywords):
        txt = txt2.replace("'s ", " 's ").replace(",", " , ").replace("-", " - ").replace(":", " : ").lower().split()
        for idx_of, word in enumerate(txt):
          matched = False
          if word in ontology:
            if word not in ignore_generic:
              txt[idx_of] = ontology[word]
            matched=True
          elif len(word) > 3 and word.endswith("s"):
            if word[:-1] in ontology:  
              if word[:-1] not in ignore_generic:
                txt[idx_of] = ontology[word[:-1]]+"s"
              matched=True
            elif word[:-2] in ontology:
              if word[:-2] not in ignore_generic:
                txt[idx_of] = ontology[word[:-2]]+"s"
              matched=True
          if matched and idx_of >0:
            prev_word = txt[idx_of-1]
            if len(prev_word) < 3 or prev_word in stopwords_hash or prev_word.endswith("ed") or prev_word.endswith("ing") or prev_word.endswith("es") or prev_word.endswith("ous") or prev_word.endswith("al") or prev_word.endswith("ly"):
              continue
            txt[idx_of-1] = ''
        txt = [t for t in txt if t]

        matched = True
        last_match_idx = None
        for times in range(5):
          if not matched:
            break
          seen={}
          matched = False
          last_match_idx = None
          for idx_of, word in enumerate(txt):
            # see if this word is a generic word
            t = word
            word2 =  ontology.get(t, ontology.get(t+"s", ontology.get(t+"es",  ontology.get(t[:-1], ontology.get(t[:-2])) if len(t) >3 and t.endswith("s") else None))) 
            # collapse a generic_word, aka Y => Y
            if idx_of > 3 and ((txt[idx_of-1]== "as" and txt[idx_of-2] in ("known", "to")) or idx_of > 2 and (txt[idx_of-1] in ("called", "including", "aka"))):
              if word2 is None:
                if last_match_idx is not None and last_match_idx >= idx_of-4:
                  for j in range(last_match_idx, idx_of):
                    txt[j]= ''
                  matched=True
                  last_matched_idx = None
              else:
                start = max(0,idx_of-3)
                start_idx = None
                generic_txt_seg = [ontology.get(t, ontology.get(t+"s", ontology.get(t+"es", ontology.get(t[:-1], ontology.get(t[:-2])) if len(t) >3 and t.endswith("s") else t))) for t in txt[start:idx_of-1]]
                if word2 in generic_txt_seg:
                  start_idx = generic_txt_seg.index(word2)+start 
                if word2+"s" in generic_txt_seg:
                  start_idx = generic_txt_seg.index(word2+"s")+start 
                elif len(word2) > 3 and word2.endswith("s") and word2[:-1] in generic_txt_seg:
                  start_idx = generic_txt_seg.index(word[:-1])+start
                elif len(word2) > 3 and word2.endswith("s") and word2[:-2] in generic_txt_seg:
                  start_idx = generic_txt_seg.index(word[:-2])+start
                elif word2 == "person" and "people" in generic_txt_seg:
                  start_idx = generic_txt_seg.index("people")+start 
                if start_idx is not None:
                  for j in range(start_idx, idx_of):
                    txt[j]= ''
                  matched=True
                  last_matched_idx = idx_of
            if word2 is not None:
              last_match_idx = idx_of
            # collapse multiple item to plural
            if idx_of > 1 and ((word2 is not None) or not (len(word) < 3 or word in stopwords_hash or word.endswith("ed") or word.endswith("ing") or word.endswith("es") or word.endswith("ous") or word.endswith("al") or word.endswith("ly"))):
              t = txt[idx_of-1]
              prev_word2 = t if t in ("and", "or", ",", "-", ":", "a", "an", "the") else ontology.get(t, ontology.get(t+"s", ontology.get(t+"es",  ontology.get(t[:-1], ontology.get(t[:-2])) if len(t) >3 and t.endswith("s") else None))) 
              if (prev_word2 in ("a", "an", "the") and idx_of > 2 and txt[idx_of-2] in ("and", "or", ",", "-", ":")) or prev_word2 in ("and", "or", ",", "-", ":") or prev_word2 == word2: 
                if (prev_word2 in ("a", "an", "the") and idx_of > 2 and txt[idx_of-2] in ("and", "or", ",", "-", ":")):
                  cc = txt[idx_of-2]
                else:
                  cc = txt[idx_of-1]
                start = max(0,idx_of-3)
                generic_txt_seg = [ontology.get(t, ontology.get(t+"s", ontology.get(t+"es",  ontology.get(t[:-1], ontology.get(t[:-2])) if len(t) >3 and t.endswith("s") else t))) for t in txt[start:idx_of]]
                if word2 is None:
                  word2 = word
                start_idx=None
                if word2 in generic_txt_seg:
                    start_idx = generic_txt_seg.index(word2)+start 
                if word2+"s" in generic_txt_seg:
                    start_idx = generic_txt_seg.index(word2+"s")+start 
                elif len(word2) > 3 and word2.endswith("s") and word2[:-1] in generic_txt_seg:
                    start_idx = generic_txt_seg.index(word[:-1])+start
                elif len(word2) > 3 and word2.endswith("s") and word2[:-2] in generic_txt_seg:
                    start_idx = generic_txt_seg.index(word[:-2])+start
                elif word2 == "person" and "people" in generic_txt_seg:
                    start_idx = generic_txt_seg.index("people")+start 
                if start_idx is not None:
                  for j in range(start_idx, idx_of):
                    txt[j] = ''
                  if cc in ("and", "or") and start > 0 and txt[start_idx-1] == ",":
                    txt[start_idx-1] = cc
                  if word2 == "person": 
                    txt[idx_of]= "people"
                  else:
                    txt[idx_of]= word2 + "s"
                  if idx_of > 2 and txt[idx_of-3] in ("a", "an"):
                    txt[idx_of-3] = "the"
                  elif idx_of > 3 and txt[idx_of-4] in ("a", "an"):
                    txt[idx_of-4] = "the"
                  matched=True
                  last_matched_idx = idx_of
            # match articles to nouns
            if idx_of > 0:
              if (word in seen or word+"s" in seen or word+"es" in seen) and txt[idx_of-1] in ("a", "an"):
                txt[idx_of-1] = "another"
                matched=True
              elif txt[idx_of-1] == "a" and word[0] in "aeiou":
                txt[idx_of-1] = "an"
                matched=True
                last_matched_idx = idx_of
              elif txt[idx_of-1] == "an" and word[0] not in "aeiou":
                txt[idx_of-1] = "a"
                matched=True
                last_matched_idx = idx_of
            # plural and singular
            if idx_of > 0 and word.endswith ("s") and txt[idx_of-1] in ("an", "a", "another"):
              if txt[idx_of-1]=="another":
                txt[idx_of-1] = "other"
              else:
                txt[idx_of-1] = "the"
              matched=True  
              last_matched_idx = idx_of
            elif idx_of > 1 and word.endswith ("s") and txt[idx_of-2] in ("an", "a", "another") and len(txt[idx_of-1]) < 4:
                if txt[idx_of-2]=="another":
                  txt[idx_of-2] = "other"
                else:
                  txt[idx_of-2] = "the" 
                matched=True 
                last_matched_idx = idx_of
            if word2 is not None:
              seen[word] = 1
          txt = [t for t in txt if t]

        txt = ' '.join([t for t in txt if t])
        return txt

def create_ontology_trim(vocab, ontology_inv):

  words = list(vocab.items())
  words.sort(key=lambda a:a[1], reverse=True)
  top2percent = words[:int(len(words)*.02)]
  top2percent = dict(top2percent)
  top25percent = words[:int(len(words)*.25)]
  top25percent = dict(top25percent)
  import copy
  top = {}
  for item in ontology_inv.items():
      for word in item[1]:
          if word in top2percent or word+"s" in top2percent or word+"es" in top2percent:
              top[word] = top2percent.get(word, top2percent.get(word+"s", top2percent.get(word+"es", 1))) 
              print (item[0], '**', word)
  for word in ontology_inv.keys():
          if word in top2percent or word+"s" in top2percent or word+"es" in top2percent:
              top[word] = top2percent.get(word, top2percent.get(word+"s", top2percent.get(word+"es", 1))) 
              print ('**', word)

  for key in list(ontology_inv.keys()):
      for word in ontology_inv[key]:
          if not (word in top25percent or word+"s" in top25percent or word+"es" in top25percent):
              print ("++", word)
      ontology_inv[key] = list(set([key]+[word for word in ontology_inv[key]  if word in top25percent or word+"s" in top25percent or word+"es" in top25percent]))

  print (top)

  ontology_inv

def load_cc(max_rng=None):
  url_to_data = {}
  txt_to_data = {}
  labels = []
  labels_end =[]
  dat = [tuple(d.strip().split("\t")) for d in open("/content/drive/MyDrive/wiki_images/ccaption.tsv").readlines()]
  dat2 = []
  out_file = open("caption2caption.tsv", "w")
  i = 0
  for txt, url, idx in dat:
    i += 1
    if max_rng is not None and i > max_rng:
      break
    
    #txt processing
    txt = txt.strip (".").strip().replace("  ", " ").strip()
    url = unquote(url)
    url = url.replace("\n", ". ")
    url = url.replace("\r", "")
    url = url.replace("\t", ". ")
    do_ignore = False
    for pat in ignore:
      if pat+" " in txt or " "+pat in txt:
        do_ignore = True
        break
    if do_ignore: continue
    for pat in pats:
      txt = txt.replace(pat+" ", "")
      txt = txt.replace(" "+pat, "")
      if not txt:
        break
    for pat in replace_lst:
      txt = txt.replace(pat[0], pat[1])
      if not txt:
        break
    txt = txt.strip (" .-,:;")
    txt = txt.split()
    if not txt: continue
    if len(txt) > 2 and txt[0] in ("d", "a", "an", "the") and txt[1] in ("a", "an", "above", "the", "of", "in", "during"):
      txt = txt[1:]
    if not txt: continue
    while txt[-1] in ("stock", "id"):
      txt = txt[:-2]
      if not txt: break
    if not txt: continue
    if len(txt) > 3 and txt[0] in ("person",) and txt[1] in ("of", "person", "during", "at", "in", "a", "an", "the", "with", "and", "or", ",") and txt[2] in ("person", ):
      if txt[1] in ("with", "and", "or", ","):
        txt = txt[3:]
        txt[0] = "people"
      else:
        txt = txt[3:]
    if not txt: continue
    while txt[0] in ("d", "of", "with", "to", "in", "during", "about"):
      txt = txt[1:]
      if not txt: break
    if not txt: continue
    #TODO cleanup garbage numbers at the bginning of the sentence
    txt = [t for t in txt if t]

    #txt = txt.replace(" - ", "-")

    #txt2 processing
    txt2 = url.split("/")[-1].split(".")[0].split("?")[0].replace("-", " ").replace("+", " ").replace("_", " ")
    
    #.strip("1234567890 ").replace("stock vector ", "").replace("stock photo ", "").replace("photo of ", "").replace("picture id", "").replace("photograph of", "").replace("photograph", "").replace("vector", "").replace("image", "").replace("illustration", "").replace("drawing", "").replace("cartoon", "")
    #TODO - keep common numbers

    txt2 = [t for t in txt2.split() if t.lower() == "3d" or len(t) < 5 or ("0" not in t and "1" not in t and "2" not in t and "3" not in t and "4" not in t and "5" not in t and "6" not in t and "7" not in t and "8" not in t and "9" not in t and len(t) < 20 and t.upper() != t)]
    if txt2 and "a" not in txt2[-1].lower() and "e" not in txt2[-1].lower() and "i" not in txt2[-1].lower() and "o" not in txt2[-1].lower() and "u" not in txt2[-1].lower(): 
      if txt2[-1].lower() not in ("gym", "sky", "why", "mtv", "nyc", "nv", "ny", "lynch", ):
        #print ("**", txt2[-1])
        txt2 = txt2[:-2]
    while txt2 and txt2[-1].lower() in ("copy", "the", "a", "an", "on", "of", "at", "during", "between", "near", "by", "in", "with", "his", "hers"):
      txt2 = txt2[:-2]
    while txt2 and txt2[0].lower() in ("of", "by", "0", ":", "-", ","):
      txt2 = txt2[1:]
    if len(txt2) <= 2:
      txt2 = txt
    test_junk = "".join(txt2)
    len_txt2 = len(test_junk)
    non_junk = "".join([a for a in txt2 if len(a)>=3]).replace("0","").replace("1","").replace("2","").replace("3","").replace("4","").replace("5","").replace("6","").replace("7","").replace("8","").replace("9","")
    len_non_junk = len(non_junk)
    if len_non_junk == 0 or len_non_junk/len_txt2 < 0.65:
      txt2 = txt
    if txt2[0] == "Abandoned":
      print (txt2)
    txt2 = " ".join(txt2)
    do_ignore = False
    for pat in ignore:
      if pat+" " in txt2:
        do_ignore = True
        break
    if do_ignore: 
      txt2 = " ".join(txt)
    #for d in del_list:
    #  txt = txt.replace(d+" ", "")
    for pat in pats:
      txt2 = txt2.replace(pat+" ", "")
      txt2 = txt2.replace(" "+pat, "")
      if not txt2:
        break
    for pat in replace_lst:
      txt2 = txt2.replace(pat[0], pat[1])
      if not txt2:
        break
    txt2 = txt2.strip (" .-,:;")
    txt2 = txt2.split()
    if not txt2: continue
    if len(txt2) > 2 and txt2[0] in ("d", "a", "an", "the") and txt2[1] in ("a", "an", "above", "the", "of", "in", "during"):
      txt2 = txt2[1:]
    if not txt2: continue
    while txt2[-1] in ("stock", "id"):
      txt2 = txt2[:-2]
      if not txt2: break
    if not txt2: continue
    if len(txt2) > 3 and txt2[0] in ("person",) and txt2[1] in ("of", "person", "during", "at", "in", "a", "an", "the", "with", "and", "or", ",") and txt2[2] in ("person", ):
      if txt2[1] in ("with", "and", "or", ","):
        txt2 = txt2[3:]
        txt2[0] = "people"
      else:
        txt2 = txt2[3:]
    if not txt2: continue
    while txt2[0] in ("d", "of", "with", "to", "in", "during", "about"):
      txt2 = txt2[1:]
      if not txt2: break
    if not txt2: continue
      #'person , person',
      # person of person
    txt2 = [t for t in txt2 if t]
    if len(txt2) < 3: 
      txt2 = txt
    is_dup = tuple(txt) != tuple(txt2)
    all_words_caption.extend([a for a in txt  if not a in stopwords_hash and len(a) > 2])
    if not is_dup:
      all_words_caption.extend([a for a in txt2  if not a in stopwords_hash and len(a) > 2])

    for idx_of, keyword in enumerate(txt2):
      lst2 = None
      if keyword in ontology or keyword in ontology_inv :
          lst2 = keywordLst2Hash[keyword] = keywordLst2Hash.get(keyword, [])
      if  lst2 is not None and len(keyword) > 4 and keyword.endswith("s"):
          if keyword[:-2] in ontology or keyword[:-2] in ontology_inv: 
            keyword = keyword[:-2]
            lst2 = keywordLst2Hash[keyword] = keywordLst2Hash.get(keyword, [])
          elif keyword[:-3] in ontology or keyword[:-3] in ontology_inv:
            keyword = keyword[:-3]
            lst2 = keywordLst2Hash[keyword] = keywordLst2Hash.get(keyword, [])
      if lst2 is not None:
          if idx_of > 1:

            lst2.extend(txt2[idx_of-2:idx_of])
          keyword = ontology.get(keyword, keyword)
          if keyword not in ignore_category:
            #print (keyword)
            lst = keywordLstHash[keyword] = keywordLstHash.get(keyword, [])
            lst.extend(txt)
            if not is_dup:
              lst.extend(txt2)

    txt = " ".join(txt)
    txt2 = " ".join(txt2)
    txt2 = txt2.replace(" s ", " 's ")
    if txt2.startswith("actress") and txt.startswith("actor"):
      txt = txt.replace("actor", "actress")
    txt = txt.strip (" .-,:;")
    txt2 = txt2.strip (" .-,:;")

    # replace unnatural labels in the generic descriptions in txt
    txt, txt2 = replace_labels(txt, txt2)

    if txt.endswith(" image"): txt = txt.replace(" image", "")
    if txt2.endswith(" image"): txt = txt2.replace(" image", "")
    if txt.endswith(" by person"): txt = txt.replace(" by person", "")
    if txt2.endswith(" by person"): txt = txt2.replace(" by person", "")


    # genericize txt2 if txt and txt2 are the same
    if txt == txt2:
      txt = txt2.replace("'s ", " 's ").replace(",", " , ").replace("-", " - ").replace(":", " : ").replace("- -", "--").replace("0 , ", "0,").replace("1 , ", "1 ,").replace("2 , ", "2,").replace("3 , ", "3,").replace("4 , ", "4,").replace("5 , ", "5,").replace("6 , ", "6,").replace("7 , ", "7,").replace("8 , ", "8,").replace("9 , ", "9,").lower().split()
      for idx_of, word in enumerate(txt):
        matched = False
        prev_word = prev_prev_word = ""
        if idx_of >0:
            prev_word = txt[idx_of-1]
        if idx_of >1:
            prev_prev_word = txt[idx_of-2]

        if word in ontology:
          if word not in ignore_generic and len(word) > 3:
              if ontology[word] == "person":
                  if (idx_of<=0 or not intersect([prev_word, prev_prev_word], ("their", "our", "your", "his", "her", "its", "my", "'s"))):
                      txt[idx_of] = ontology[word]
              else:
                  txt[idx_of] = ontology[word]
          matched=True
        elif len(word) > 3 and word.endswith("s"):
          if word[:-1] in ontology:  
            if word[:-1] not in ignore_generic and len(word) > 3:
              if ontology[word[:-1]] == "person":
                  if (idx_of<=0 or not intersect([prev_word, prev_prev_word], ("their", "our", "your", "his", "her", "its", "my", "'s"))):
                      txt[idx_of] = "people"
              elif ontology[word[:-1]] in ("waters","food", "clothes", "kitchenware", "military"):
                txt[idx_of] = ontology[word[:-1]]
              else:
                txt[idx_of] = ontology[word[:-1]]+"s"
            matched=True
          elif word[:-2] in ontology:
            if word[:-2] not in ignore_generic and len(word) > 3:
              if ontology[word[:-2]] == "person":
                  if (idx_of<=0 or not intersect([prev_word, prev_prev_word], ("their", "our", "your", "his", "her", "its", "my", "'s"))):
                      txt[idx_of] = "people"
              elif ontology[word[:-2]] in ("waters","food", "clothes", "kitchenware", "military"):
                txt[idx_of] = ontology[word[:-2]]
              else:
                txt[idx_of] = ontology[word[:-2]]+"s"
            matched=True
      txt = [t for t in txt if t]

      matched = True
      last_match_idx = None
      for times in range(5):
        if not matched:
          break
        seen={}
        matched = False
        last_match_idx = None
        for idx_of, word in enumerate(txt):
          # see if this word is a generic word
          t = word
          word2 =  ontology.get(t, ontology.get(t+"s", ontology.get(t+"es",  ontology.get(t[:-1], ontology.get(t[:-2])) if len(t) >3 and t.endswith("s") else None))) 
          # collapse a generic_word, aka Y => Y
          if idx_of > 3 and ((txt[idx_of-1]== "as" and txt[idx_of-2] in ("known", "to")) or idx_of > 2 and (txt[idx_of-1] in ("called", "including", "aka"))):
            if word2 is None:
              if last_match_idx is not None and last_match_idx >= idx_of-4:
                for j in range(last_match_idx, idx_of):
                  txt[j]= ''
                matched=True
            else:
              start = max(0,idx_of-3)
              start_idx = None
              generic_txt_seg = [ontology.get(t, ontology.get(t+"s", ontology.get(t+"es", ontology.get(t[:-1], ontology.get(t[:-2])) if len(t) >3 and t.endswith("s") else t))) for t in txt[start:idx_of-1]]
              if word2 in generic_txt_seg:
                start_idx = generic_txt_seg.index(word2)+start 
              if word2+"s" in generic_txt_seg:
                start_idx = generic_txt_seg.index(word2+"s")+start 
              elif len(word2) > 3 and word2.endswith("s") and word2[:-1] in generic_txt_seg:
                start_idx = generic_txt_seg.index(word[:-1])+start
              elif len(word2) > 3 and word2.endswith("s") and word2[:-2] in generic_txt_seg:
                start_idx = generic_txt_seg.index(word[:-2])+start
              elif word2 == "person" and "people" in generic_txt_seg:
                start_idx = generic_txt_seg.index("people")+start 
              if start_idx is not None:
                for j in range(start_idx, idx_of):
                  txt[j]= ''
                matched=True
          # remove descriptors before generic word
          if word2 is not None and idx_of >0:
            prev_prev_word = prev_prev_prev_word = ""
            prev_word = txt[idx_of-1]
            if idx_of >1:
              prev_prev_word = txt[idx_of-2]
              if idx_of > 2:
                prev_prev_prev_word = txt[idx_of-3]
              else:
                prev_prev_prev_word = prev_prev_word
            if  idx_of <= 2 or intersect([prev_prev_word, prev_prev_prev_word],("with", "of", "on", "in", "during", "at", "by", "a", "an", "this", "that",  "some", "the", "all", "other", "another", "those", "their", "our", "your", "his", "her", "'s", "its", "my")):
              if not (prev_prev_word  in ("least", "most") or prev_prev_prev_word in ("least", "most") or len(prev_word) < 3 or prev_word in stopwords_hash  or prev_word.endswith("ed") or prev_word.endswith("ing") or prev_word.endswith("es") or prev_word.endswith("ous")): # or prev_word.endswith("al") or prev_word.endswith("less") or prev_word.endswith("able") or prev_word.endswith("ful") or prev_word.endswith("ly")
                txt[idx_of-1] = ''
                matched=True
          # collapse multiple item to plural
          if idx_of > 1 and ((word2 is not None) or not (len(word) < 3 or word in stopwords_hash or word.endswith("ed") or word.endswith("ing") or word.endswith("es") or word.endswith("ous") or word.endswith("al") or word.endswith("less") or word.endswith("able") or word.endswith("ful") or word.endswith("ly"))):
            t = txt[idx_of-1]
            prev_word2 = t if t in ("and", "or", ",", "a", "an", "the") else ontology.get(t, ontology.get(t+"s", ontology.get(t+"es",  ontology.get(t[:-1], ontology.get(t[:-2])) if len(t) >3 and t.endswith("s") else None))) 
            if (prev_word2 in ("a", "an", "the") and idx_of > 2 and txt[idx_of-2] in ("and", "or", ",",)) or prev_word2 in ("and", "or", ",",) or prev_word2 == word2: 
              if (prev_word2 in ("a", "an", "the") and idx_of > 2 and txt[idx_of-2] in ("and", "or", ",")):
                cc = txt[idx_of-2]
              else:
                cc = txt[idx_of-1]
              start = max(0,idx_of-3)
              generic_txt_seg = [ontology.get(t, ontology.get(t+"s", ontology.get(t+"es",  ontology.get(t[:-1], ontology.get(t[:-2])) if len(t) >3 and t.endswith("s") else t))) for t in txt[start:idx_of]]
              if word2 is None:
                word2 = word
              start_idx=None
              if word2 in generic_txt_seg:
                  start_idx = generic_txt_seg.index(word2)+start 
              if word2+"s" in generic_txt_seg:
                  start_idx = generic_txt_seg.index(word2+"s")+start 
              elif len(word2) > 3 and word2.endswith("s") and word2[:-1] in generic_txt_seg:
                  start_idx = generic_txt_seg.index(word[:-1])+start
              elif len(word2) > 3 and word2.endswith("s") and word2[:-2] in generic_txt_seg:
                  start_idx = generic_txt_seg.index(word[:-2])+start
              elif word2 == "person" and "people" in generic_txt_seg:
                  start_idx = generic_txt_seg.index("people")+start 
              if start_idx is not None:
                for j in range(start_idx, idx_of):
                  txt[j] = ''
                if cc in ("and", "or") and start > 0 and txt[start_idx-1] == ",":
                  txt[start_idx-1] = cc
                if word2 == "person": 
                  txt[idx_of]= "people"
                elif word2 not in ("waters", "clothes", "food", "kitchenware", "military"):
                  txt[idx_of]= word2 + "s"
                if idx_of > 2 and txt[idx_of-3] in ("a", "an"):
                  txt[idx_of-3] = "the"
                elif idx_of > 3 and txt[idx_of-4] in ("a", "an"):
                  txt[idx_of-4] = "the"
                matched=True

          # match articles to nouns
          if idx_of > 0:
            if (word in seen or word+"s" in seen or word+"es" in seen) and txt[idx_of-1] in ("a", "an"):
              txt[idx_of-1] = "another"
              matched=True
            elif txt[idx_of-1] in ("my", "your", "our", "their") and word2 not in ("person", "people") and not intersect([word], ("hand", "feet", "foot", "wrist", "face", "ear", "hair", "nose", "back", "leg", "arm", "legs", "arms", "chest", "eye", "eyes")):
              txt[idx_of-1] = "the"
              matched=True
            elif txt[idx_of-1] == "a" and word[0] in "aeiou":
              txt[idx_of-1] = "an"
              matched=True
            elif txt[idx_of-1] == "an" and word[0] not in "aeiou":
              txt[idx_of-1] = "a"
              matched=True
          # plural and singular
          if idx_of > 0 and (word in ("waters", "clothes", "food", "people", "kitchenware", "military") or word.endswith ("s")) and txt[idx_of-1] in ("an", "a", "another"):
            if txt[idx_of-1]=="another":
              txt[idx_of-1] = "other"
            else:
              txt[idx_of-1] = "the"
            matched=True  
          elif idx_of > 1 and (word in ("waters", "clothes", "food", "people", "kitchenware", "military") or word.endswith ("s")) and txt[idx_of-2] in ("an", "a", "another") and len(txt[idx_of-1]) < 4:
              if txt[idx_of-2]=="another":
                txt[idx_of-2] = "other"
              else:
                txt[idx_of-2] = "the" 
              matched=True 
 
          # remove common adjetives and adverbs
          if word2 is None and len(word)>4 and word not in stopwords_hash and (word.endswith("al") or word.endswith("less") or word.endswith("able") or word.endswith("ful") or word.endswith("ly")):
              if idx_of <= 0 or txt[idx_of-1] not in ("and", "or", ",", "look", "looks", "looked", "most",):
                  txt[idx_of] = "" 
                  matched=True 
              
          if word2 is not None:
              last_matched_idx = idx_of
              seen[word] = 1
        txt = [t for t in txt if t]

      txt = ' '.join([t for t in txt if t])
      if txt != txt2:
          tmp = txt2
          txt2 = txt.strip("., ")
          txt = tmp.strip("., ")
          #print ("generized **", txt, '**', txt2)
          pass

    # Test "Abandonned"
    txt_collapsed = "".join([a[:3] for a in txt.replace(",", "").replace(".", "").replace("-", "").replace(":", "").split() if len(a)>=3]).lower()
    txt2_collapsed = "".join([a[:3] for a in txt2.replace(",", "").replace(".", "").replace("-", "").replace(":", "").split() if len(a)>=3]).lower()
    if txt_collapsed == txt2_collapsed or txt_collapsed in txt2_collapsed or txt2_collapsed in txt_collapsed:
        if len(txt) > len(txt2):
          txt2 = txt
        else:
          txt = txt2

    txt2_lower = txt2.lower()
    if txt2_lower in txt_to_data:
      data = txt_to_data[txt2_lower]
      txt_set = data[0]
      data[0] = list(set(txt_set+[txt2, txt]))
    else:
      txt_to_data[txt2_lower] = [list(set([txt2, txt])), [url], [idx]]
  return txt_to_data

def collapse_lines(params, lines, input_file, output_file):
  # continuous collapsing of similar sentences
  collapsed = False
  prev_txt2 = ""
  for line in lines:
      txt2, data = line.split("\t")
      if prev_txt2:
        prev_txt2_collapsed = "".join([a[:3] for a in prev_txt2.split() if len(a)>=3])
        txt2_collapsed = "".join([a[:3] for a in txt2.split() if len(a)>=3])
        if prev_txt2_collapsed.lower() == txt2_collapsed.lower():
          txt_arr =prev_data2[0]
          txt_arr2 = [txt_arr+prev_data2[0]]
          txt_arr2.remove(txt_arr[0])
          txt_arr2 = [txt_arr[0]] + txt_arr2
          prev_data2[0] = txt_arr2
          collapsed = True
          continue      
      prev_txt2 = txt2
      prev_data2 = data
  if not collapsed:
    return False
  return True

def url_to_data(args, lines, inputfile, outpufile):
  # test if urls are the same
  for item in lines:
    item = item.split("\t")
    txt2 = items[0]
    data = items[1]
    txt_lst = data[0]
    urls = data[1]
    for url in urls:
      url2 = "http:"+url.lower().replace("https:", "http:").split("http:")[-1]
      #for ty in (".jpg", ".png", ".gif", ".tiff", ".jpeg"):
      #  if ty in url:
      #    url2 = url.split(ty, 1)[0] + ty
      #    break
      if url2 in url_to_data:
        data = url_to_data[url2]
        txt_set = data[0]
        data[0] = list(set(txt_set+[txt2]))
        print (url2, data)
      else:
        url_to_data[url2]= [[txt2]]

  for dataU in url_to_data.values():
    data = None
    for txt in dataU[0]:
      if data is not None:
        dataO = txt_to_data[txt]
        for txt2 in dataO[0]:
          if txt2 not in data[0]:
            data[0].append(txt2)
        for url2 in dataO[1]:
          if url2 not in data[1]:
            data[1].append(url2)
        for idx2 in dataO[2]:
          if idx2 not in data[2]:
            data[2].append(idx2)
        del txt_to_data[txt]
      else:
        data = txt_to_data[txt]
  url_to_data=None
  
  for data in txt_to_data.values():
    for tl in data[0]:
      labels.append(" ".join(tl.strip().split()[:3]))
      labels_end.append(" ".join(tl.strip().split()[-3:]))
    txt_lst = data[0]
    url = data[1]
    idx = data[2]
    out_file.write(txt_lst[0] + "\t" + ";#;".join(txt_lst) + "\t" + ";#;".join(url) + "\t"+ ";#;".join(idx)+ "\n")
  
  c = Counter(labels)
  c2 =Counter(labels_end)
  return c, c2

def intersect(a, b):
  if not a or not b:
    return False
  if len(a) < len(b):
    c = a
    a = b
    b = c
  if type(b) is dict:
    c = a
    a = b
    b = c
  if a is not dict:
    a = dict([(a1.lower() if len(a1) < 4 else a1[:4].lower(), 1) for a1 in a])
  for b1 in b:
    b1 = b1.lower() if len(b1) < 4 else b1[:4].lower()
    if b1 in a:
      return True
  return False
  
def replace_labels(txt, txt2):

  is_dup  = txt == txt2
  txt = txt.split()
  txt2 = txt2.split()
  len_txt2 = len(txt2)
  if (len_txt2 >3 and "biological" in txt2[:3]) or (len_txt2 > 2 and "biological" in txt2[:2]) or  (len_txt2 > 1 and "biological" in txt2[:1]) or ("biological" == txt2[0]):
    c = txt
    txt = txt2
    txt2 = c
  len_txt = len(txt)
  if txt[0] == "biological":
    txt[0] = "an"
    txt[1] = "animal"
  elif len_txt > 1 and txt[1] in ("biological",):
    txt = txt[1:]
    txt[0] = "an"
    txt[1] = "animal"
  elif len_txt > 2 and txt[2] in ("biological",):
    txt = txt[2:]
    txt[0] = "an"
    txt[1] = "animal"
  elif len_txt > 3 and txt[3] in ("biological",):
    txt = txt[3:]
    txt[0] = "an"
    txt[1] = "animal"
  if len(txt) >= 3 and txt[1] == "animal":
    if intersect([txt2[0], txt2[1]], ("man", "woman", "child", "men", "women", "children", "boy", "girl", "boys", "girls", "kid", "kids")):
      txt[0] = "a"
      txt[1] = "person"
    if len(txt) > 2 and intersect([txt[2]], ("plant", "animal", "black", "white", "silhouette", "red", "filming", "of", "up", "out", "a", "an", "the", "this", "what", "their", "there", "eggs", "morning", "man", "woman", "child", "men", "women", "children", "boy", "girl", "boys", "girls", "kid", "kids")):
      txt = txt[2:]
    if len(txt) > 3 and intersect([txt[3]],("a", "an", "the", "this", "that", "those", "his", "hers", "one")) and intersect([txt[2]], ("was", "were", "is", "are", ":", "-")):
      txt = txt[3:]
    if len(txt) > 3 and intersect([txt[3]],  ("toast", "bacon", "person", "plant", "plants", )) and intersect([txt[2]],  ("and", "with" )):
      txt = txt[3:]
      #and "town" not in txt 
    txt3 = txt + txt2
    if  intersect(([txt[2]] if len(txt) >2  else []) +  ([txt2[2]] if len(txt2) >2  else []), 
                  ("hide", "hiding", "walking", "peering", "peer", "peers", "awake", "awakened", "looking", "looks", \
                       "look", "gathers", "gathering", "gather", "chew", "chewing", "chews", "feed", "feeds", "feeding", "foraging", \
                       "sit", "sits", "sitting", "perch", "perches", "perched", "perching", "collect", "collects", "collecting", "eat",\
                       "eating", "eats", "fly", "flies", "flying", "arrive", "attend" )) or \
                       intersect(txt3, ("bombus", "hummingbird",  "ants", "baby", "aphids",  "chameleon",  "mantis",  "permiere", "his", "her",  "dog",  "dogs",  "frog", "butterfly",  "bee", "bees",  "honeybee")) or \
                       (not intersect(txt3, plant_list) and not intersect(txt3, food_list)) or \
                       intersect(txt3, animal_list):
        #pass
 #         (not intersect(txt2[:2], ('tree', 'onions', 'ivy', 'moss',  'rose', 'wisteria', 'avocado', 'broccoli', 'pineapple', 'barley', 'maize', 'oak', 'papaver',  'lily', 'basil', \
 #                     'parsley','bellis', 'poison', 'begonia','garden', 'elm', 'birch', 'lotus', 'clover', 'caraway', 'cotton', 'dried')) and \

        if len(txt) > 2 and intersect([txt[2]], ("plant", "animal", "black", "white", "silhouette", "red", "filming", "of", "up", "out", "a", "an", "the", "this", "what", "their", "there", \
                                                "eggs", "morning", "man", "woman", "child", "men", "women", "children", "boy", "girl", "boys", "girls", "kid", "kids")):
            txt = txt[2:]

        if is_dup:
          txt2 = txt
        else:
          if intersect([txt2[0]], ("many", "several", "many", "some", "all", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")):
            txt[0] = txt2[0]
            txt[1] = txt[1]+"s"
          if  (txt2[0] not in ("breasted", ) and txt2[0].endswith("ed")) or  txt2[0].endswith("ful") or txt2[0].endswith("able") or txt2[0].endswith("ly") or txt2[0].endswith("ous") or intersect([txt2[0]], plant_animal_descriptor):
            txt[0] = txt2[0]
          elif (txt2[1] not in ("breasted", ) and txt2[1].endswith("ed")) or txt2[1].endswith("ful") or txt2[1].endswith("able") or txt2[1].endswith("ly") or txt2[1].endswith("ous") or intersect([txt2[1]], plant_animal_descriptor):
            txt[0] = txt2[1]      
    else:
        if len(txt) > 1:
          txt[0] = "a"
          txt[1] = "plant"
          if len(txt) > 2 and intersect([txt[2]], ("leaf", "leaves", "trunk", "tree", "flower", "flowers")):
            if len(txt) > 3 and intersect([txt[3]],  ("his", "hers", "the", "a")):
              txt[1] = "person"
            else:
              txt[1] = "plant's"
        if is_dup:
          txt2 = txt
        else:
          if intersect([txt2[0]], ("many", "several", "many", "some", "all", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")):
            txt[0] = txt2[0]
            txt[1] = txt[1]+"s"
          if  (txt2[0] not in ("breasted", ) and txt2[0].endswith("ed")) or  txt2[0].endswith("ful") or txt2[0].endswith("able") or txt2[0].endswith("ly") or txt2[0].endswith("ous") or intersect([txt2[0]], plant_animal_descriptor):
            txt[0] = txt2[0]
          elif (txt2[1] not in ("breasted", ) and txt2[1].endswith("ed")) or txt2[1].endswith("ful") or txt2[1].endswith("able") or txt2[1].endswith("ly") or txt2[1].endswith("ous") or intersect([txt2[1]], plant_animal_descriptor):
            txt[0] = txt2[1]
  is_food = intersect(txt+txt2, food_list)
  is_bird = intersect(txt+txt2, ("bird", "feather", "nest", "flock", "flies", "fly", "flying", "flight", "perch", "chirp", "swoop", "soar", "wing"))
  is_plant = "needles" in txt or "needles" in txt2 or intersect(txt+txt2, plant_list)
  is_team = intersect(txt+txt2,("game", "ball", "play",))
  txt=" ".join(txt)
  txt2 = " ".join(txt2)
  if is_food:
        txt = txt.replace("biological kingdom", "food")
        txt2 = txt2.replace("biological kingdom", "food")
        txt = txt.replace("biological subspecies", "food")
        txt2 = txt2.replace("biological subspecies", "food")
        txt = txt.replace("biological species", "food")
        txt2 = txt2.replace("biological species", "food")
        txt = txt.replace("biological genus", "food")
        txt2 = txt2.replace("biological genus", "food")
        txt = txt.replace("a plant", "food")
        txt2 = txt2.replace("a plant", "food")
        txt = txt.replace("an animal", "food")
        txt2 = txt2.replace("an animal", "food")
  elif is_plant:
        if "plant" in txt or "plant" in txt2:
          det = "another"
        else:
          det = "a"
        txt = txt.replace("biological kingdom", det+" plant")
        txt2 = txt2.replace("biological kingdom", det+" plant")
        txt = txt.replace("biological subspecies", det+" plant")
        txt2 = txt2.replace("biological subspecies", det+" plant")
        txt = txt.replace("biological species", det+" plant")
        txt2 = txt2.replace("biological species", det+" plant")
        txt = txt.replace("biological genus", det+" plant")
        txt2 = txt2.replace("biological genus", det+" plant")
  elif is_team:
        if "team" in txt or "team" in txt2:
          det = "another"
        else:
          det = "a"
        txt = txt.replace("biological kingdom", det+" team")
        txt2 = txt2.replace("biological kingdom", det+" team")
        txt = txt.replace("biological subspecies", det+" team")
        txt2 = txt2.replace("biological subspecies", det+" team")
        txt = txt.replace("biological species", det+" team")
        txt2 = txt2.replace("biological species", det+" team")
        txt = txt.replace("biological genus", det+" team")
        txt2 = txt2.replace("biological genus", det+" team")  
        txt = txt.replace("plant", "team")
        txt2 = txt2.replace("plant", "team")  
        txt = txt.replace("animal", "team")
        txt2 = txt2.replace("animal", "team")  
  elif is_bird:
        if "team" in txt or "team" in txt2:
          det = "another"
        else:
          det = "a"
        txt = txt.replace("biological kingdom", det+" bird")
        txt2 = txt2.replace("biological kingdom", det+" bird")
        txt = txt.replace("biological subspecies", det+" bird")
        txt2 = txt2.replace("biological subspecies", det+" bird")
        txt = txt.replace("biological species", det+" bird")
        txt2 = txt2.replace("biological species", det+" bird")
        txt = txt.replace("biological genus", det+" bird")
        txt2 = txt2.replace("biological genus", det+" bird") 
        txt = txt.replace("animal", "bird")
        txt2 = txt2.replace("animal", "bird") 
        txt = txt.replace("plant", "bird")
        txt2 = txt2.replace("plant", "bird") 
  else:
        if "animal" in txt or "animal" in txt2:
          det = "another"
        else:
          det = "a"
        txt = txt.replace("biological kingdom", det+" animal")
        txt2 = txt2.replace("biological kingdom", det+" animal")
        txt = txt.replace("biological subspecies", det+" animal")
        txt2 = txt2.replace("biological subspecies", det+" animal")
        txt = txt.replace("biological species", det+" animal")
        txt2 = txt2.replace("biological species", det+" animal")
        txt = txt.replace("biological genus", det+" animal")
        txt2 = txt2.replace("biological genus", det+" animal")  
  return (txt, txt2)    


def process_cc(args, lines, inputfile, outpufile):
  c, c2 = load_cc()
  ll = list(c.items())
  ll.sort(key=lambda a: a[1], reverse=True)
  lle = list(c2.items())
  lle.sort(key=lambda a: a[1], reverse=True)

