from timm.models import create_model
from utils.models.model_factory_32 import parse_params
from utils_architecture import get_new_model

def build_model(model_name, num_classes, model_params=None, **kwargs):
    model_name = model_name.lower()
    model_config = parse_params(model_params)
    print(f'Model params: {model_config}')
    
    if model_name == 'sslresnext50':
        model_name = 'SSLResNext50'
        model = create_model('ssl_resnext50_32x4d', num_classes=num_classes, **model_config)
    elif model_name == 'resnet50':
        model_name = 'ResNet50'
        model = create_model('resnet50', num_classes=num_classes, **model_config)
    elif model_name == 'wide_resnet50_2':
        model_name = 'WideResNet_50_2'
        model = create_model('wide_resnet50_2', num_classes=num_classes, **model_config)
    elif model_name == 'wide_resnet50_4':
        model_name = 'WideResNet_50_4'
        # wide_resnet50_4 is not defined in system timm, so create it manually with base_width=256
        from timm.models.resnet import ResNet, Bottleneck, _create_resnet
        model_args = dict(block=Bottleneck, layers=(3, 4, 6, 3), base_width=256)
        model = _create_resnet('wide_resnet50_4', pretrained=False, num_classes=num_classes, **dict(model_args, **model_config))
    elif model_name == 'tresnetm':
        model_name = 'TResNet-M'
        model = create_model('tresnet_m', num_classes=num_classes, **model_config)
    elif model_name == 'seresnext26t':
        model_name = 'SE-ResNeXt-26-T'
        model = create_model('seresnext26t_32x4d', num_classes=num_classes, **model_config)
    elif model_name == 'seresnext50':
        model_name = 'SE-ResNeXt-50'
        model = create_model('seresnext50_32x4d', num_classes=num_classes, **model_config)
    else:
        assert 'convnext' in model_name.lower()
        # For models in utils_architecture.py, only support num_classes=1000
        assert num_classes == 1000, f'Model {model_name} from utils_architecture only supports num_classes=1000, got {num_classes}'
        
        print(f'Using model {model_name} from utils_architecture.py')
        model = get_new_model(model_name, pretrained=True, not_original=True)
        model_name = model_name.replace('_', '-').title()

    config = dict(name=model_name, **model_config)
    return model, model_name, config, 224
