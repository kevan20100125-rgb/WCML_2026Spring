import json
from importlib_resources import files
import numpy as np
import torch
from opencood.models.torch_tdl import models

SPEED_OF_LIGHT=torch.tensor(299792458)
PI=torch.tensor(3.141592653589793)

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
def uniform_rand(size,left,right,dtype):
    #print(tuple(size))
    raw_rand=torch.rand(tuple(size),dtype=dtype)
    uniform_rand_=(raw_rand*(right-left))+left
    #print(uniform_rand_.shape)
    return uniform_rand_

class TDL(torch.nn.Module):
    def __init__(self,model,
                    delay_spread,
                    carrier_frequency,
                    num_sinusoids=20,
                    los_angle_of_arrival=PI/4.,
                    min_speed=0.,
                    max_speed=None,
                    num_rx_ant=1,
                    num_tx_ant=1,
                    spatial_corr_mat=None,
                    rx_corr_mat=None,
                    tx_corr_mat=None,
                    dtype=torch.complex64):
        super(TDL, self).__init__()
        self._dtype = dtype
        real_dtype = torch.float32
        self._real_dtype = real_dtype
        self._num_rx_ant = num_rx_ant
        self._num_tx_ant = num_tx_ant
        if model == 'A':
            parameters_fname = "TDL-A.json"
        elif model == 'B':
            parameters_fname = "TDL-B.json"
        elif model == 'C':
            parameters_fname = "TDL-C.json"
        elif model == 'D':
            parameters_fname = "TDL-D.json"
        elif model == 'E':
            parameters_fname = "TDL-E.json"
        elif model == 'A30':
            parameters_fname = "TDL-A30.json"
            if delay_spread != 30e-9:
                print("Warning: Delay spread is set to 30ns with this model")
                delay_spread = 30e-9
        elif model == 'B100':
            parameters_fname = "TDL-B100.json"
            if delay_spread != 100e-9:
                print("Warning: Delay spread is set to 100ns with this model")
                delay_spread = 100e-9
        elif model == 'C300':
            parameters_fname = "TDL-C300.json"
            if delay_spread != 300e-9:
                print("Warning: Delay spread is set to 300ns with this model")
                delay_spread = 300e-9

        # Load model parameters
        self._load_parameters(parameters_fname)

        self._carrier_frequency = torch.tensor(carrier_frequency)
        self._num_sinusoids = torch.tensor(num_sinusoids)
        self._los_angle_of_arrival = los_angle_of_arrival
        self._delay_spread = torch.tensor(delay_spread)
        self._min_speed = torch.tensor(min_speed)
        if max_speed is None:
            self._max_speed = self._min_speed
        else:
            assert max_speed >= min_speed, \
                "min_speed cannot be larger than max_speed"
            self._max_speed = torch.tensor(max_speed)
        # Pre-compute maximum and minimum Doppler shifts
        self._min_doppler = self._compute_doppler(self._min_speed)
        self._max_doppler = self._compute_doppler(self._max_speed)
        alpha_const = 2. * PI / num_sinusoids * \
                      torch.arange(1., self._num_sinusoids + 1, 1.)

        self._alpha_const = torch.reshape(alpha_const,
                                       [1,  # batch size
                                        1,  # num rx
                                        1,  # num rx ant
                                        1,  # num tx
                                        1,  # num tx ant
                                        1,  # num clusters
                                        1,  # num time steps
                                        num_sinusoids])

        self._spatial_corr_mat_sqrt = None
        self._rx_corr_mat_sqrt = None
        self._tx_corr_mat_sqrt = None


    def forward(self, batch_size, num_time_steps, sampling_frequency):
        # Time steps
        sample_times = torch.arange(num_time_steps, dtype=self._real_dtype) \
                       / sampling_frequency
        sample_times = torch.unsqueeze(insert_dims(sample_times, 6, 0), -1)
        doppler = uniform_rand([batch_size,
                                     1,  # num rx
                                     1,  # num rx ant
                                     1,  # num tx
                                     1,  # num tx ant
                                     1,  # num clusters
                                     1,  # num time steps
                                     1],  # num sinusoids
                                    self._min_doppler,
                                    self._max_doppler,
                                    self._real_dtype)
        theta = uniform_rand([batch_size,
                                   1,  # num rx
                                   1,  # 1 RX antenna
                                   1,  # num tx
                                   1,  # 1 TX antenna
                                   self._num_clusters,
                                   1,  # num time steps
                                   self._num_sinusoids],
                                  -PI / torch.tensor(self._num_sinusoids,
                                                dtype=self._real_dtype),
                                  PI / torch.tensor(self._num_sinusoids,
                                                    dtype=self._real_dtype),
                                  self._real_dtype)
        alpha = self._alpha_const + theta
        phi = uniform_rand([batch_size,
                                 1,  # 1 RX
                                 self._num_rx_ant,  # 1 RX antenna
                                 1,  # 1 TX
                                 self._num_tx_ant,  # 1 TX antenna
                                 self._num_clusters,
                                 1,  # Phase shift is shared by all time steps
                                 self._num_sinusoids],
                                -PI,
                                PI,
                                self._real_dtype)
        argument = doppler * sample_times * torch.cos(alpha) + phi
        h = torch.complex(torch.cos(argument), torch.sin(argument))
        normalization_factor = 1. / torch.sqrt(self._num_sinusoids.type(
                                                    self._real_dtype))
        h = torch.complex(normalization_factor, torch.tensor(0.,dtype=self._real_dtype)) \
            * torch.sum(h, axis=-1)
        #print(h.shape)
        # Scaling by average power
        mean_powers = torch.unsqueeze(insert_dims(self._mean_powers, 5, 0), -1)
        h = torch.sqrt(mean_powers) * h

        if self._los:
            phi_0=uniform_rand([batch_size,
                                   1,  # num rx
                                   1,  # 1 RX antenna
                                   1,  # num tx
                                   1,  # 1 TX antenna
                                   1,
                                   1],  # num time steps],
                                  -PI,
                                   PI,
                                   self._real_dtype)
            doppler=torch.squeeze(doppler,dim=-1)
            sample_times = torch.squeeze(sample_times, dim=-1)
            #doppler=doppler.reshape(batch_size,1,1,1,1,1,1)
            #sample_times=sample_times.reshape(batch_size,1,1,1,1,1,1)
            arg_spec=doppler*sample_times*torch.cos(self._los_angle_of_arrival)+phi_0
            h_spec=torch.complex(torch.cos(arg_spec),torch.sin(arg_spec))
            #print(h_spec.shape,h.shape)
            #print((h_spec*torch.sqrt(self._los_power)+h[:,:,:,:,:,:1,:]).shape)
            h=torch.concat([h_spec*torch.sqrt(self._los_power)+h[:,:,:,:,:,:1,:],h[:,:,:,:,:,1:,:]],dim=-2)
            #print(h.shape)

        # Delays
        if self._scale_delays:
            delays = self._delays * self._delay_spread
        else:
            delays = self._delays * 1e-9  # ns to s
        delays = insert_dims(delays, 3, 0)
        delays = delays.repeat(batch_size, 1, 1, 1)
        h.requires_grad=False
        delays.requires_grad=False
        return  h, delays


    def _load_parameters(self,fname):
        source = files(models).joinpath(fname)
        # pylint: disable=unspecified-encoding
        with open(source) as parameter_file:
            params = json.load(parameter_file)

        # LoS scenario ?
        self._los = bool(params['los'])

        # Scale the delays
        self._scale_delays = bool(params['scale_delays'])

        # Loading cluster delays and mean powers
        num_clusters=params['num_clusters']
        #print(num_clusters)
        self._num_clusters = num_clusters

        # Retrieve power and delays
        delays=params['delays']
        delays = torch.tensor(delays)
        mean_powers = np.power(10.0, np.array(params['powers']) / 10.0)
        mean_powers = torch.tensor(mean_powers)

        if self._los:
            # The power of the specular component of the first path is stored
            # separately
            self._los_power = mean_powers[0]
            mean_powers = mean_powers[1:]
            # The first two paths have 0 delays as they correspond to the
            # specular and reflected components of the first path.
            # We need to keep only one.
            delays = delays[1:]

        # Normalize the PDP
        if self._los:
            norm_factor = torch.sum(mean_powers) + self._los_power
            self._los_power = self._los_power / norm_factor
            mean_powers = mean_powers / norm_factor
        else:
            norm_factor = torch.sum(mean_powers)
            mean_powers = mean_powers / norm_factor

        self._delays = delays
        self._mean_powers = mean_powers

    def _compute_doppler(self, speed):
        return 2. * PI * speed / SPEED_OF_LIGHT * self._carrier_frequency





