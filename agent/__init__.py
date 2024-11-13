from agent import hiql, qrl

algos = {
    "hiql": hiql.create_learner,
    "qrl": qrl.create_learner,
}
