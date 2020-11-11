import torch
import torch.nn as nn

label = 0

# Connected Network -2 Decision Tree        https://github.com/AaronX121/Soft-Decision-Tree/
class Net_DT(nn.Module):

    def __init__(self, input_dim, output_dim, depth=5, lamda=1e-3, use_cuda=False):
        super(Net_DT, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.lamda = lamda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self._validate_parameters()
        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        # Different penalty coefficients for nodes in different layers
        self.penalty_list = [self.lamda * (2 ** (-depth))
                             for depth in range(0, self.depth)]

        # Initialize internal nodes and leaf nodes, the input dimension on
        # internal nodes is added by 1, serving as the bias.
        self.inner_nodes = nn.Sequential(
            nn.Linear(self.input_dim + 1,
                      self.internal_node_num_, bias=False),
            nn.Sigmoid())

        self.leaf_nodes = nn.Linear(self.leaf_node_num_,
                                    self.output_dim, bias=False)

    def forward(self, X, is_training_data=False):

        _mu, _penalty = self._forward(X)
        y_pred = self.leaf_nodes(_mu)

        # When `X` is the training data, the model also returns the penalty
        # to compute the training loss.
        if is_training_data:
            return y_pred, _penalty
        else:
            return y_pred

    def _cal_penalty(self, layer_idx, _mu, _path_prob):

        penalty = torch.tensor(0.).to(self.device)

        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))

        for node in range(0, 2 ** (layer_idx + 1)):
            alpha = (torch.sum(_path_prob[:, node] * _mu[:, node // 2], dim=0) /
                     torch.sum(_mu[:, node // 2], dim=0))

            layer_penalty_coeff = self.penalty_list[layer_idx]

            penalty -= 0.5 * layer_penalty_coeff * (torch.log(alpha) +
                                                    torch.log(1 - alpha))

        return penalty

    """ 
      Add a constant input `1` onto the front of each instance. 
    """

    def _data_augment(self, X):
        batch_size = X.size()[0]
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        X = torch.cat((bias, X), 1)

        return X

    def _validate_parameters(self):

        if not self.depth > 0:
            msg = 'The tree depth should be strictly positive, but got {} instead.'
            raise ValueError(msg.format(self.depth))

        if not self.lamda >= 0:
            msg = ('The coefficient of the regularization term should not be'
                   ' negative, but got {} instead.')
            raise ValueError(msg.format(self.lamda))
