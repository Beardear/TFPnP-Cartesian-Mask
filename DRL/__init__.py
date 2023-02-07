from DRL.actor import ResNetActor
from DRL.ddpg_csmri import DDPG_CSMRI


def get_actor(opt):
    # CS-MRI
    action_bundle = opt.action_pack
    C = 6
    if opt.solver == 'admm':
        actor = ResNetActor(C+3, action_bundle)
    else:
        raise NotImplementedError

    return actor
