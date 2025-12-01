import numpy as np

class EnsembleDeepRVFL:
    """A ensemble deep RVFL classifier or regression.
    Attributes:
        n_nodes: An integer of enhancement node number.
        lam: A floating number of regularization parameter.
        w_random_vec_range: A list, [min, max], the range of generating random weights.
        b_random_vec_range: A list, [min, max], the range of generating random bias.
        random_weights: A Numpy array shape is [n_feature, n_nodes], weights of neuron.
        random_bias: A Numpy array shape is [n_nodes], bias of neuron.
        beta: A Numpy array shape is [n_feature + n_nodes, n_class], the projection matrix.
        activation: A string of activation name.
        n_layer: A integer, N=number of hidden layers.
        data_std: A list, store normalization parameters for each layer.
        data_mean: A list, store normalization parameters for each layer.
        same_feature: A bool, the true means all the features have same meaning and boundary for example: images.
        task_type: A string of ML task type, 'classification' or 'regression'.
    """
    def __init__(self, n_nodes=40, lam=1, w_random_vec_range=[-10, 10], b_random_vec_range=[0, 10], activation='relu', n_layer=10, random_seed: int=1, same_feature=False,
                 task_type='regression'):
        assert task_type in ['classification', 'regression'], 'task_type should be "classification" or "regression".'
        self.n_nodes = n_nodes
        self.lam = lam
        self.w_random_range = w_random_vec_range
        self.b_random_range = b_random_vec_range
        
        self.activation = activation
        self.get_activation()
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(random_seed)

        self.n_layer = n_layer
        
        self.same_feature = same_feature

        self.task_type = task_type

        self.random_weights = []
        self.random_bias = []
        self.beta = []
        self.data_std = [None] * self.n_layer
        self.data_mean = [None] * self.n_layer
    
    def get_activation(self):
        a = Activation()
        self.activation_function = getattr(a, self.activation)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Args:
            deep: boolean, optional
                If True, will return the parameters for this estimator and
                contained subobjects that are estimators.
        
        Returns:
            params: mapping of string to any
                Parameter names mapped to their values.
        """
        return {"n_nodes": self.n_nodes, "lam": self.lam, "w_random_vec_range": self.w_random_range,
                "b_random_vec_range": self.b_random_range, "activation": self.activation, "n_layer": self.n_layer,
                "random_seed": self.random_seed, "same_feature": self.same_feature}

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Args:
            **params: Dictionary of parameter names mapped to their values
        
        Returns:
            self: returns an instance of self.
        """
        for parameter, value in params.items():
            setattr(self, parameter, value)
        self.get_activation()
        self.random_state = np.random.RandomState(self.random_seed)
        self.random_weights = []
        self.random_bias = []
        self.beta = []
        self.data_std = [None] * self.n_layer
        self.data_mean = [None] * self.n_layer
        return self
    
    def fit(self, data, label, n_class=None):
        """
        :param data: Training data.
        :param label: Training label.
        :param n_class: An integer of number of class. In regression, this parameter won't be used.
        :return: No return
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        assert len(label.shape) == 1, 'Label should be 1-D array.'

        n_sample = len(data)
        n_feature = len(data[0])
        h = data.copy()
        data = self.standardize(data, 0)
        if self.task_type == 'classification':
            y = self.one_hot(label, n_class)
        else:
            y = label
        for i in range(self.n_layer):
            h = self.standardize(h, i)  # Normalization data
            self.random_weights.append(self.get_random_vectors(len(h[0]), self.n_nodes, self.w_random_range))
            self.random_bias.append(self.get_random_vectors(1, self.n_nodes, self.b_random_range))
            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, data], axis=1)

            h = d

            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

            if n_sample > (self.n_nodes + n_feature):
                self.beta.append(np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y))
            else:
                self.beta.append(d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y))

    def predict(self, data, output_prob=False):
        """
        :param data: Predict data.
        :return: When classification, return vote result,  addition result and probability.
                 When regression, return the mean output of edrvfl.
        """
        n_sample = len(data)
        h = data.copy()
        data = self.standardize(data, 0)  # Normalization data
        outputs = []
        for i in range(self.n_layer):
            h = self.standardize(h, i)  # Normalization data
            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, data], axis=1)

            h = d

            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
            outputs.append(np.dot(d, self.beta[i]))
        if self.task_type == 'classification':
            vote_res = [np.argmax(item, axis=1) for item in outputs]
            vote_res = list(map(np.bincount, list(np.array(vote_res).transpose())))
            vote_res = np.array(list(map(np.argmax, vote_res)))

            add_proba = self.softmax(np.sum(outputs, axis=0))
            add_res = np.argmax(add_proba, axis=1)
            return vote_res, (add_res, add_proba)
        elif self.task_type == 'regression':
            return np.mean(outputs, axis=0)


    def eval(self, data, label):
        """
        :param data: Evaluation data.
        :param label: Evaluation label.
        :return: When classification return vote and addition accuracy.
                 When regression return MAE.
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        assert len(label.shape) == 1, 'Label should be 1-D array.'

        n_sample = len(data)
        h = data.copy()
        data = self.standardize(data, 0)
        outputs = []
        for i in range(self.n_layer):
            h = self.standardize(h, i)  # Normalization data

            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, data], axis=1)

            h = d

            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

            outputs.append(np.dot(d, self.beta[i]))
        if self.task_type == 'classification':
            vote_res = [np.argmax(item, axis=1) for item in outputs]
            vote_res = list(map(np.bincount, list(np.array(vote_res).transpose())))
            vote_res = np.array(list(map(np.argmax, vote_res)))
            vote_acc = np.sum(np.equal(vote_res, label)) / len(label)

            add_proba = self.softmax(np.sum(outputs, axis=0))
            add_res = np.argmax(add_proba, axis=1)
            add_acc = np.sum(np.equal(add_res, label)) / len(label)

            return vote_acc, add_acc
        elif self.task_type == 'regression':
            pred = np.mean(outputs, axis=0)
            mae = np.mean(np.abs(pred - label))
            return mae

    def get_random_vectors(self, m, n, scale_range):
        x = (scale_range[1] - scale_range[0]) * self.random_state.random([m, n]) + scale_range[0]
        return x

    @staticmethod
    def one_hot(x, n_class):
        y = np.zeros([len(x), n_class])
        for i in range(len(x)):
            y[i, x[i]] = 1
        return y

    def standardize(self, x, index):
        if self.same_feature is True:
            if self.data_std[index] is None:
                self.data_std[index] = np.maximum(np.std(x), 1/np.sqrt(len(x)))
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x)
            return (x - self.data_mean[index]) / self.data_std[index]
        else:
            if self.data_std[index] is None:
                self.data_std[index] = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x, axis=0)
            return (x - self.data_mean[index]) / self.data_std[index]

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.repeat((np.sum(np.exp(x), axis=1))[:, np.newaxis], len(x[0]), axis=1)


class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** (-x))

    @staticmethod
    def sine(x):
        return np.sin(x)

    @staticmethod
    def hardlim(x):
        return (np.sign(x) + 1) / 2

    @staticmethod
    def tribas(x):
        return np.maximum(1 - np.abs(x), 0)

    @staticmethod
    def radbas(x):
        return np.exp(-(x**2))

    @staticmethod
    def sign(x):
        return np.sign(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def leaky_relu(x):
        x[x >= 0] = x[x >= 0]
        x[x < 0] = x[x < 0] / 10.0
        return x