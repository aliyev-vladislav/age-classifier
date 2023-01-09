from transformers import ViTFeatureExtractor, ViTForImageClassification, pipeline

def initialize():
    name = "nateraw/vit-age-classifier"
    extractor = ViTFeatureExtractor.from_pretrained(name)
    model = ViTForImageClassification.from_pretrained(name)
    return pipeline("image-classification", model=model, feature_extractor=extractor)

def classify(self, image):
    return self(image)
