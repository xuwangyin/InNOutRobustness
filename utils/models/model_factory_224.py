from timm.models import create_model
from utils.models.model_factory_32 import parse_params

def build_model(model_name, num_classes, model_params=None, **kwargs):
    model_name = model_name.lower()
    model_config = parse_params(model_params)
    print(f'Model params: {model_config}')

    if model_name == 'sslresnext50':
        model_name = 'SSLResNext50'
        model = create_model('ssl_resnext50_32x4d', num_classes=num_classes, **model_config)
    elif model_name.startswith('resnet50'):
        model_name = 'ResNet50'
        model = create_model('resnet50', num_classes=num_classes, **model_config)
    elif model_name == 'wide_resnet50_2':
        model_name = 'WideResNet_50_2'
        model = create_model('wide_resnet50_2', num_classes=num_classes, **model_config)
    elif model_name == 'tresnetm':
        model_name = 'TResNet-M'
        model = create_model('tresnet_m', num_classes=num_classes, **model_config)
    elif model_name == 'seresnext26t':
        model_name = 'SE-ResNeXt-26-T'
        model = create_model('seresnext26t_32x4d', num_classes=num_classes, **model_config)
    elif model_name == 'seresnext50':
        model_name = 'SE-ResNeXt-50'
        model = create_model('seresnext50_32x4d', num_classes=num_classes, **model_config)
    elif model_name == 'convnext_tiny':
        model_name = 'ConvNeXt-Tiny'
        model = create_model('convnext_tiny', num_classes=num_classes, **model_config)
    elif model_name == 'convnext_small':
        model_name = 'ConvNeXt-Small'
        model = create_model('convnext_small', num_classes=num_classes, **model_config)
    elif model_name == 'convnext_base':
        model_name = 'ConvNeXt-Base'
        model = create_model('convnext_base', num_classes=num_classes, **model_config)
    elif model_name == 'convnext_large':
        model_name = 'ConvNeXt-Large'
        model = create_model('convnext_large', num_classes=num_classes, **model_config)
    else:
        print(f'Net {model_name} not supported')
        raise NotImplemented()

    config = dict(name=model_name, **model_config)
    return model, model_name, config, 224
