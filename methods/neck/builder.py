import methods


def build_neck(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]

    neck = methods.neck.__dict__[cfg['type']](**param)

    return neck
