import methods


def build_model(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]

    model = methods.__dict__[cfg.type](**param)

    return 