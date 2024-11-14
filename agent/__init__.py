from agent import hiql, qrl, crl

algos = {
    "hiql": hiql.create_learner,
    "qrl": qrl.create_learner,
    "crl": crl.create_learner,
}
