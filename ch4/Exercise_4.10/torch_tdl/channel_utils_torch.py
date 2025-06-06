import torch
from opencood.models.torch_tdl.TDL_torch import PI


def insert_dims(tensor, num_dims, axis=-1):
    rank = tensor.ndim
    #print(rank)
    axis = axis if axis >= 0 else rank + axis + 1
    #print(axis)
    shape = tensor.shape
   # print(shape)
   # print(shape[:axis],shape[axis:])
    new_shape = torch.cat([torch.tensor(shape[:axis]),
                           torch.ones((int(num_dims)), dtype=torch.int32),
                           torch.tensor(shape[axis:])], axis=0).int()
  #  print(tuple(new_shape.numpy()))
    output = torch.reshape(tensor,tuple(new_shape.numpy()))
   # print(output.shape)

    return output

def expand_to_rank(tensor, target_rank, axis=-1):
    num_dims = max_num(target_rank - tensor.ndim, 0)
    output = insert_dims(tensor, num_dims, axis)

    return output
def max_num(a,b):
    if a>b:
        return a
    else:
        return b

def uniform_rand(size,left,right,dtype):
    #print(tuple(size))
    raw_rand=torch.rand(tuple(size),dtype=dtype)
    uniform_rand_=(raw_rand*(right-left))+left
    #print(uniform_rand_.shape)
    return uniform_rand_

def subcarrier_frequencies(num_subcarriers, subcarrier_spacing):


    if  int(num_subcarriers)%2== 0:
        start=-num_subcarriers/2
        limit=num_subcarriers/2
    else:
        start=-(num_subcarriers-1)/2
        limit=(num_subcarriers-1)/2+1

    frequencies = torch.arange(int(start*2),
                            int(limit*2),
                            step=2)/2
    frequencies = frequencies*subcarrier_spacing
    return frequencies

def cir_to_ofdm_channel(frequencies, a, tau, normalize=False):

    real_dtype = tau.dtype
    if len(tau.shape) == 4:
        # Expand dims to broadcast with h. Add the following dimensions:
        #  - number of rx antennas (2)
        #  - number of tx antennas (4)

        tau = torch.unsqueeze(torch.unsqueeze(tau, dim=2), dim=4)
        # Broadcast is not supported yet by TF for such high rank tensors.
        # We therefore do part of it manually
        tau = tau.repeat(1, 1, 1, 1, a.shape[4], 1)

    # Add a time samples dimension for broadcasting
    tau = torch.unsqueeze(tau, axis=6)

    # Bring all tensors to broadcastable shapes
    tau = torch.unsqueeze(tau, axis=-1)
    h = torch.unsqueeze(a, axis=-1)
    frequencies = expand_to_rank(frequencies, tau.ndim, axis=0)
    #print(frequencies.shape)
    ## Compute the Fourier transforms of all cluster taps
    # Exponential component
    e = torch.exp(torch.complex(torch.tensor(0, dtype=real_dtype),
                                -2 * PI * frequencies * tau))

    #print(e.shape)
    #print(h.shape)
    h_f = h*e
    # Sum over all clusters to get the channel frequency responses
    h_f = torch.sum(h_f, axis=-3)

    if normalize:
        # Normalization is performed such that for each batch example and
        # link the energy per resource grid is one.
        # Average over TX antennas, RX antennas, OFDM symbols and
        # subcarriers.
        c = torch.mean(torch.square(torch.abs(h_f)), dim=(2, 4, 5, 6), keepdim=True)
        c = torch.sqrt(c)
        h_f = torch.div(h_f, c)

    return h_f

def complex_normal(shape, var=1.0, dtype=torch.complex64):
    # Half the variance for each dimension
    var_dim = torch.tensor(var)/torch.tensor(2)
    stddev = torch.sqrt(var_dim)

    # Generate complex Gaussian noise with the right variance
    xr = torch.randn(shape, dtype=torch.float32) * stddev
    xi = torch.randn(shape, dtype=torch.float32) * stddev
    x = torch.complex(xr, xi)

    return x

def flatten_last_dims(tensor, num_dims=2):
    if num_dims == len(tensor.shape):
        new_shape = [-1]
        return tensor.reshape(new_shape)
    else:
        tensor_reshape=torch.flatten(tensor,start_dim=-num_dims,end_dim=-1)
        return tensor_reshape

