def clip_image_to_multitext_score(clip_model, clip_processor, image, text_array, clip_vision_output=None, decompose_image=False):
  p = next(clip_model.parameters())
  decomposed_image_features = None 
  if clip_vision_output is None:
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
      inputs['pixel_values'] = inputs['pixel_values'].to(dtype=p.dtype, device=p.device)
      inputs['return_dict'] = True
      clip_vision_output = clip_model.vision_model(**inputs)
      if decompose_image:
        o = (clip_output["last_hidden_state"] + 4*clip_vision_output["last_hidden_state"][:,0,:])/5
        clip_vision_output.decomposed_image_features = clip_model.visual_projection(clip_model.vision_model.post_layernorm(o))
      clip_vision_output.image_features = clip_model.visual_projection(clip_vision_output["pooler_output"])
  image_features = clip_vision_output.image_features
  if hasattr(clip_vision_output, 'decomposed_image_features'):
    decomposed_image_features = clip_vision_output.decomposed_image_features
  inputs = clip_processor(text_array, padding=True, return_tensors="pt").to(p.device)
  with torch.no_grad():
    text_features = clip_model.get_text_features(**inputs)
  scores =  cosine_similarity(image_features, text_features, dim=1)
  if decompose_image:
    decomposed_scores_topk = []
    decomposed_scores = []
    for tfeat in text_features:
      scores2 =  cosine_similarity(decomposed_image_features.squeeze(0), tfeat.unsqueeze(0), dim=1)
      decomposed_scores_topk.append(scores2.topk(k=decomposed_image_features.shape[1]))
      decomposed_scores.append(decomposed_scores_topk[-1].values[0])
  else:
    decomposed_scores_topk = [] 
    decomposed_scores = []
  return {'scores': scores, 'decomposed_scores': torch.stack(decomposed_scores) if decomposed_scores else [], 'decomposed_scores_topk': decomposed_scores_topk, 'clip_vision_output': clip_vision_output, 'text_features': text_features}
