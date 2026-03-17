import torch
import torch.nn as nn
import torch.nn.functional as F
from toolbox.backbone.CLIP.clip import clip

import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
try:
    response = urllib.request.urlopen("https://example.com")
    print(response.read())
except Exception as e:
    print(e)


CATEGORIES = ["imp.surf", "building", "low veg.", "tree", "car"]
PROMPT_TEMPLATES_SAMPLE = "a photo of a {category}"
PROMPT_TEMPLATES = {
    "regular": "a photo of a regular {category}",
    "linear": "a photo of a linear {category}",
    "regional": "a photo of a regional {category}"
}
#subtype
template = "Aerial view of {}"

class_definitions = {
    "imp.surf": [
        "asphalt road or highway",
        "concrete pavement or sidewalk",
        "parking lot surface"
    ],
    "building": [
        "concrete commercial building roof",
        "residential house roof",
        "industrial warehouse building"
    ],
    "low veg.": [
        "green grassland or lawn",
        "agricultural crop field",
        "low bushes and shrubs"
    ],
    "tree": [
        "dense forest canopy",
        "green tree crown",
        "clusters of trees"
    ],
    "car": [
        "small passenger vehicle",
        "large truck or van",
        "vehicles in a parking lot"
    ],

    # P

    # "Clutter": [
    #     "dark building shadow on pavement or grass",
    #     "tree canopy shadow on ground surface",
    #     "small urban water body: pond, stream, or fountain"
    # ]
}

class CLIPS(torch.nn.Module):
    def __init__(self, clip_model="ViT-B/32"):

        super(CLIPS, self).__init__()

        self.model, _ = clip.load(clip_model)

    def build_subtype_features(self):

        all_phrases = []

        with torch.no_grad():
            for class_name, subtypes in class_definitions.items():

                texts = [template.format(subtype) for subtype in subtypes]

                tokens = clip.tokenize(texts).cuda()

                class_feats = self.model.encode_text(tokens)

                class_feats = class_feats / class_feats.norm(dim=-1, keepdim=True)
                mean_feat = class_feats.mean(dim=0)
                mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)

                all_phrases.append(mean_feat)
        self.text_subtype = torch.stack(all_phrases).float()

    def build_sample(self):
        all_phrases = []

        for category in CATEGORIES:
            phrase = PROMPT_TEMPLATES_SAMPLE.format(category=category)
            all_phrases.append(phrase)
        text = clip.tokenize(all_phrases).cuda()
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_sample = text_features
    def build_clip_text_features(self):
        all_phrases = []

        for category in CATEGORIES:
            for prompt_type in ["regular", "linear", "regional"]:
                phrase = PROMPT_TEMPLATES[prompt_type].format(category=category)
                all_phrases.append(phrase)

        text_tokens = clip.tokenize(all_phrases).cuda()
        text_features = []

        for i in range(text_tokens.size(0)):
            with torch.no_grad():
                feature = self.model.encode_text(text_tokens[i].unsqueeze(0))
            feature = feature / feature.norm(dim=-1, keepdim=True)
            feature = feature.to(torch.float32)
            text_features.append(feature)
        text_features = torch.cat(text_features, dim=0)

        category_features = []
        for i in range(len(CATEGORIES)):
            cls_features = text_features[i * 3: i * 3 + 3]
            cls_avg = torch.mean(cls_features, dim=0, keepdim=True)
            cls_avg = cls_avg.float()
            category_features.append(cls_avg)
        self.category_features = torch.cat(category_features, dim=0)
        self.category_features = self.category_features/self.category_features.norm(dim=-1, keepdim=True)
        self.category_features = self.category_features.to(torch.float32)

if __name__ == "__main__":
    with torch.no_grad():
        model = CLIPS().cuda()
        model.build_clip_text_features()
        print(model.category_features.dtype)

