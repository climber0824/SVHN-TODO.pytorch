class TrainingConfig(object):
    Batch_Size = 32
    Learning_Rate = 1e-2
    StepsToCheckLoss = 100
    StepsToSnapshot = 10000
    StepsToDecay = 10000
    DecayRate = 0.9
    StepsToFinish = 3000000


class TestingConfig(object):
    Batch_Size = 256
