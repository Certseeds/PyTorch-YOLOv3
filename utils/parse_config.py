def add_init(module_defs):
    module_defs.insert(1, {})
    module_defs[1]["type"] = "expandlayer"
    module_defs[1]["filters"] = 3
    module_defs.insert(1, {})
    module_defs[1]["type"] = "graylayer"
    module_defs[1]["filters"] = 1
    return module_defs


def parse_model_config(path: str):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    with open(path, "r") as file:
        lines = file.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        module_defs = []
        for line in lines:
            if line.startswith('['):  # This marks the start of a new block
                module_defs.append({})
                module_defs[-1]['type'] = line[1:-1].rstrip()
                if module_defs[-1]['type'] == 'convolutional':
                    module_defs[-1]['batch_normalize'] = 0
            else:
                key, value = line.split("=")
                value = value.strip()
                module_defs[-1][key.rstrip()] = value.strip()
    module_defs = add_init(module_defs)
    return module_defs


def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
