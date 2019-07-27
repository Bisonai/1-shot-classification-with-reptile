# Bison AI pretrained model
from pretrained_models.bisonai.models import OmniglotModelBisonai

model_names = {
    "bisonai" : OmniglotModelBisonai,
    }

def build_model(NAME,
            num_classes):

    model = model_names[NAME](num_classes=num_classes)
    return model
