from timm.models import create_model
from resnet import resnet50, wide_resnet50_2

def build_model(model_name, num_classes, **kwargs):
    model_name = model_name.lower()
    if model_name == 'sslresnext50':
        model_name = 'SSLResNext50'
        model = create_model('ssl_resnext50_32x4d', num_classes=num_classes, **kwargs)
        config = dict(name=model_name, **kwargs)
    elif model_name == 'resnet50':
        model_name = 'ResNet50'
        print(kwargs)
        model = resnet50(num_classes=num_classes)
        # model = create_model('resnet50', num_classes=num_classes, **kwargs)
        config = dict(name=model_name, **kwargs)
    elif model_name == 'resnet50_timm':
        model_name = 'ResNet50_timm'
        print(kwargs)
        print('using timm')
        model = create_model('resnet50', num_classes=num_classes, **kwargs)
        config = dict(name=model_name, **kwargs)
    elif model_name == 'wide_resnet50_2':
        model_name = 'WideResNet_50_2'
        print(kwargs)
        model = wide_resnet50_2(num_classes=num_classes)
        config = dict(name=model_name, **kwargs)
    elif model_name == 'wide_resnet50_2_timm':
        print('using timm')
        model_name = 'WideResNet_50_2_timm'
        model = create_model('wide_resnet50_2', num_classes=num_classes, **kwargs)
        config = dict(name=model_name, **kwargs)
    elif model_name == 'tresnetm':
        model_name = 'TResNet-M'
        model = create_model('tresnet_m', num_classes=num_classes, **kwargs)
        config = dict(name=model_name, **kwargs)
    elif model_name == 'seresnext26t':
        model_name = 'SE-ResNeXt-26-T'
        model = create_model('seresnext26t_32x4d', num_classes=num_classes, **kwargs)
        config = dict(name=model_name, **kwargs)
    elif model_name == 'seresnext50':
        model_name = 'SE-ResNeXt-50'
        model = create_model('seresnext50_32x4d', num_classes=num_classes, **kwargs)
        config = dict(name=model_name, **kwargs)
    else:
        print(f'Net {model_name} not supported')
        raise NotImplemented()

    return model, model_name, config, 224
