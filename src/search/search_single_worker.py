import warnings

import numpy as np
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Float,
    InCondition,
)
from src.search.dpn_objective import fairness_objective_dpn
from smac import MultiFidelityFacade, Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"
import time

from smac.multi_objective.parego import ParEGO

from smac.runner.dask_runner import DaskParallelRunner
import dask

if __name__ == "__main__":
    cs = ConfigurationSpace()
    #import dask
    #client = dask.distributed.Client(scheduler_file="scheduler-dpn-25-file.json")
    # Define our environment variables
    edge1 = Categorical("edge1", ["0","1","2","3","4","5","6","7","8"], default="2")
    edge2 = Categorical("edge2", ["0","1","2","3","4","5","6","7","8"], default="0")
    edge3 = Categorical("edge3", ["0","1","2","3","4","5","6","7","8"], default="1")
    lr_adam = Float("lr_adam", (1e-5, 1e-2), default=1e-3, log=True)
    lr_sgd = Float("lr_sgd", (0.09,0.8), log=True, default=0.1)
    head = Categorical("head", ["CosFace","ArcFace","MagFace"], default="CosFace")
    optimizer = Categorical("optimizer",["Adam","AdamW","SGD"], default="Adam")
    cond_1 = InCondition(lr_adam, parent = optimizer, values = ['Adam','AdamW'])
    cond_2 = InCondition(lr_sgd, parent = optimizer, values = ['SGD'])
    cs.add_hyperparameters([
        edge1,
        edge2,
        edge3,
        lr_adam,
        lr_sgd,
        optimizer,
        head,
    ])
    cs.add_conditions([cond_1,cond_2])
    objectives = ["time", "loss"]

    # Define our environment variables
    scenario = Scenario(
        cs,
        objectives=objectives,
        walltime_limit=100000000, 
        n_trials=200,  # Evaluate max 200 different trials
        min_budget=2,  
        max_budget=10,  
        n_workers=1,
    )

    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=16)
    intensifier = MultiFidelityFacade.get_intensifier(scenario, eta=2)
    multi_objective_algorithm = ParEGO(scenario)

    smac = MultiFidelityFacade(
        scenario,
        fairness_objective_dpn,
        initial_design=initial_design,
        multi_objective_algorithm=multi_objective_algorithm,
        overwrite=True,
        intensifier = intensifier,
    )
    #        target_fn=target_fn,
    # Let's optimize
    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(cs.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")
