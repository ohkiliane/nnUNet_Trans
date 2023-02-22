import torch
from nnunet.network_architecture.generic_TransUNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnTransUNetTrainerV2 import nnTransUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn

#from torch import nn

class SMU1(nn.Module):
    '''
    Implementation of SMU-1 activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha: hyper parameter
    References:
        - See related paper:
        https://arxiv.org/abs/2111.04682
    Examples:
        >>> smu1 = SMU1()
        >>> x = torch.Tensor([0.6,-0.3])
        >>> x = smu1(x)
    '''
    def __init__(self, alpha = 0.25):
        '''
        Initialization.
        INPUT:
            - alpha: hyper parameter
            aplha is initialized with zero value by default
        '''
        super(SMU1,self).__init__()
        self.alpha = alpha
        # initialize mu
        self.mu = torch.nn.Parameter(torch.tensor(4.352665993287951e-9)) 
        
    def forward(self, x):
        return ((1+self.alpha)*x+torch.sqrt(torch.square(x-self.alpha*x)+torch.square(self.mu)))/2

class nnTransUNetTrainerV2_SMU1(nnTransUNetTrainerV2):
    def initialize_network(self):
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = SMU1
        net_nonlin_kwargs = {}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes), self.patch_size,
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
