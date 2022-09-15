import os
from datasets import load_dataset
d = load_dataset("succinctly/midjourney-prompts")
txt = [a.split("--")[0] for a in d['train']['text'] if len(a) > 20]
txt = [a.replace("::", " ").replace("  ", " ").replace(" ,", ", ").split("—")[0].strip("+-.;: ").rstrip("0123456789+-.;:").lower().replace("/imagine prompt:", "").replace("/imagine", "").strip("/ ") for a in txt if a and  a[0] != "<" and len(a) > 20]
txt = [a.replace("::", " ").replace("  ", " ").replace(" ,", ", ").split("—")[0].strip("+-.;: ").rstrip("0123456789+-.;:").lower().replace("/imagine prompt:", "").replace("/imagine", "").strip("/ ") for a in txt if a and  a[0] != "<" and len(a) > 20]
txt = [a.replace("::", " ").replace("  ", " ").replace(" ,", ", ").split("—")[0].strip("+-.;: ").rstrip("0123456789+-.;:").lower().replace("/imagine prompt:", "").replace("/imagine", "").strip("/ ") for a in txt if a and  a[0] != "<" and len(a) > 20]
txt = [a.replace("::", " ").replace("  ", " ").replace(" ,", ", ").split("—")[0].strip("+-.;: ").rstrip("0123456789+-.;:").lower().replace("/imagine prompt:", "").replace("/imagine", "").strip("/ ") for a in txt if a and  a[0] != "<" and len(a) > 20]
txt = [a.replace("::", " ").replace("  ", " ").replace(" ,", ", ").split("—")[0].strip("+-.;: ").rstrip("0123456789+-.;:").lower().replace("/imagine prompt:", "").replace("/imagine", "").strip("/ ") for a in txt if a and  a[0] != "<" and len(a) > 20]
txt = [a.replace("::", " ").replace("  ", " ").replace(" ,", ", ").split("—")[0].strip("+-.;: ").rstrip("0123456789+-.;:").lower().replace("/imagine prompt:", "").replace("/imagine", "").strip("/ ") for a in txt if a and  a[0] != "<" and len(a) > 20]
txt = list(set(txt))
txt  = list(dict([(a[:20].replace(" ", "").replace(".", "").replace(",", "").replace("'", "").replace("\"", ""), a) for a in txt]).values())
txt.sort()
open("/content/drive/Shareddrives/ontocord/midjourney.txt", "w", encoding="utf8").write("\n".join(txt))
os.system("gzip /content/drive/Shareddrives/ontocord/midjourney.txt")
