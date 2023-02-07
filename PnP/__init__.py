from PnP.solver_csmri import ADMMSolver_CSMRI


def get_solver(opt):
    print('[i] use solver: {}'.format(opt.solver))
    if opt.solver == 'admm':
        solver = ADMMSolver_CSMRI(opt)
    else:
        raise NotImplementedError

    return solver

