class TrainingConfig(object):
    Batch_Size = 32
    Learning_Rate = 1e-4
    Patience = 100
    StepsToCheckLoss = 100
    StepsToValidate = 1000
    StepsToSnapshot = 1000
    StepsToDecayLearningRate = 10000
    StepsToFinish = 10000
    DecayRate = 0.9
