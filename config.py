class TrainingConfig(object):
    Batch_Size = 32
    Learning_Rate = 1e-5
    StepsToCheckLoss = 10
    StepsToSnapshot = 5000
    StepsToDecay = 30000
    StepsToFinish = 100000


class TestingConfig(object):
    Batch_Size = 256
