from model import resnetdsbn

__sets = {}

__sets[model_name] = (lambda num_classes, in_features, pretrained, num_domains, model_name=model_name)


def get_model(model_name, num_classes, in_features=0, num_domains=2, pretrained=False):
    model_key = model_name
    return __sets[model_key](num_classes=num_classes, in_features=in_features, pretrained=pretrained, num_domains=num_domains)
