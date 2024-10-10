from typing import get_args, Literal

# from models.RPCANetHQS import *


Models = Literal["rpcanet", "mynet"]

all_models = get_args(Models)


def get_model(name: Models, net=None):
    if name == "rpcanet":
        from models.RPCANet import RPCANet

        net = RPCANet(stage_num=6)
    elif name == "mynet":
        from models.RPCANetHQS import RPCANetHQS

        net = RPCANetHQS(stage_num=6)
    else:
        raise NotImplementedError

    return net
