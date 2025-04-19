import numpy as np

def load_weights(filepath):
    with open(filepath) as file:
        s = file.readline()
    weights_str = s.strip().split(', ')
    weights = [np.float32(x) for x in weights_str]
    return weights




if __name__ == '__main__':
    weights = load_weights('class_weights.txt')
    print(weights)



