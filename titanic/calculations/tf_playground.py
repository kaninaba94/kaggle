import numpy as np

def predict(x, y):
    '''
    https://playground.tensorflow.org/#activation=relu&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0.001&noise=20&networkShape=3&seed=0.70866&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
    features: x, y
    hiddens: 1 (3)
    activation: ReLu
    regularization: None
    Learning rate: 0.03
    epochs: 631
    dataset: circle
    train_perc: 50
    noise: 20
    batchsie: 10
    :param x: feature x position
    :param y: feature y position
    :return: class prediction. If > 0 -> blue  otherwise orange
    '''
    features = np.array([[x], [y]])
    w0 = np.array([[0.98, -0.017], [1.4, -1.4], [1.4, 1.3]])
    b0 = np.array([[2.4], [-1.3], [-0.85]])
    before_relu = np.add(np.matmul(w0, features), b0)
    a1 = before_relu * (before_relu > 0)
    w1 = np.array([2.2, -2.3, -1.9])
    return np.matmul(w1, a1)
