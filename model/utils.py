from . import cnn5_avgp_fc1


model_map = {
    'cnn5-avgp-fc1': {
        'net': cnn5_avgp_fc1.SNN,
    },
}


def getNetwork(model_type: str, simulation_params):
    return model_map[model_type]['net'](simulation_params)
