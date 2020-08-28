from sentimentanalyser.utils.data import compose, listify
from sentimentanalyser.utils.optimizers import maybe_update, get_defaults


class Optimizer:
    def __init__(self, params, steppers, **defaults):
        self.steppers = listify(steppers)
        maybe_update(self.steppers, defaults, get_defaults)

        self.param_groups = list(params)

        if not isinstance(self.param_groups[0], list):
            self.param_groups = [self.param_groups]

        self.hypers = [{**defaults} for p in self.param_groups]

    def grad_params(self):
        return [(p, hyper) for pg, hyper in zip(self.param_groups, self.hypers)
                for p in pg if p.grad is not None]

    def zero_grad(self):
        for p, hyper in self.grad_params():
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        for p, hyper in self.grad_params():
            compose(p, self.steppers, **hyper)


class StatefulOptimizer(Optimizer):
    def __init__(self, params, steppers, stats=None, **defaults):
        self.stats = listify(stats)
        # getting hyper-parameter default values from Stat objects.
        maybe_update(self.stats, defaults, get_defaults)
        super().__init__(params, steppers, **defaults)
        # It's going to be a dictionary of dictionaries.
        # One dict for each parameter with contains the state info
        # based on what stats need to be stored.
        self.state = {}

    def step(self):
        for p, hyper in self.grad_params():
            if p not in self.state:
                # When running for the first time.
                # Create a state for p and call all the statistics to initialize it.
                self.state[p] = {}
                # Take the default value of states from the init_state function
                # in the stat object.
                maybe_update(self.stats, self.state[p], lambda o: o.init_state(p))
            # Take the previous state.
            state = self.state[p]
            for stat in self.stats:
                # update it.
                state = stat.update(p, state, **hyper)
            # Run steppers this time also pass the state as well.
            compose(p, self.steppers, **state, **hyper)
            # set the new state as prev for next iteration.
            self.state[p] = state
