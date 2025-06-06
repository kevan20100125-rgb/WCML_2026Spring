from opencood.models.torch_tdl.channel_utils_torch import expand_to_rank,insert_dims,\
                                subcarrier_frequencies,cir_to_ofdm_channel,\
                                complex_normal
from opencood.models.torch_tdl.TDL_torch import TDL
from opencood.models.torch_tdl.TDL_torch import PI,SPEED_OF_LIGHT
import torch
import scipy.io
#print(PI)



'''
class GenerateOFDMChannel(torch.nn.Module):
    def __init__(self, resource_grid, normalize_channel=True,
                 dtype=torch.complex64):
        super(GenerateOFDMChannel, self).__init__()
        self._normalize_channel = normalize_channel
        dataset=[]
        for i in range(1,11):
            path = './channel_data/channel_los/channel_coff_los_{}.mat'.format(i)
            matr = scipy.io.loadmat(path)
            matr= torch.tensor(matr['coff']).to(torch.complex64)
            dataset.append(matr)
        dataset=torch.cat(dataset,dim=0)
        self.dataset = dataset
        #print(dataset.shape)
    def forward(self, batch_size=None):

        tap1=torch.randint(0,50,[])
        tap2 = torch.randint(0, 12601 - 1400,[])

        h_freq = self.dataset[tap1, :, tap2:tap2 + 1400]
        h_freq = h_freq.reshape(2048, 1400)
        # 计算 coff_part 中每个元素的绝对值的平方
        h_freq = h_freq.permute(1, 0)
        #h_freq=self.dataset[tap1,:,tap2:tap2+14]
        #h_freq=h_freq.permute(1,0)
        h_freq=torch.reshape(h_freq,[1,1,1,1,1,1400,2048])
        #print(h_freq)
        if self._normalize_channel:
            # Normalization is performed such that for each batch example and
            # link the energy per resource grid is one.
            # Average over TX antennas, RX antennas, OFDM symbols and
            # subcarriers.
            c = torch.mean(torch.square(torch.abs(h_freq)), dim=(2, 4, 5, 6), keepdim=True)
            c = torch.sqrt(c)
            #print(c)
            #h_freq=h_freq*0.5
            h_freq = torch.div(h_freq, c)
        #print(h_freq)

        return h_freq

'''

#print(PI)
class GenerateOFDMChannel(torch.nn.Module):
    def __init__(self, resource_grid, normalize_channel=True,
                 dtype=torch.complex64):
        super(GenerateOFDMChannel, self).__init__()
        # Callable used to sample channel input responses
        # We need those in call()
        self._num_ofdm_symbols = resource_grid.num_ofdm_symbols
        self._subcarrier_spacing = resource_grid.subcarrier_spacing
        self._num_subcarriers = resource_grid.fft_size
        self._normalize_channel = normalize_channel
        self._sampling_frequency = 1./resource_grid.ofdm_symbol_duration

        # Frequencies of the subcarriers
        self._frequencies = subcarrier_frequencies(self._num_subcarriers,
                                                   self._subcarrier_spacing)
        self._cir_sampler = TDL('E',delay_spread = 100e-9,
          carrier_frequency = 3.5e9,
          min_speed = 0.0,
          max_speed = 3.0)
    def forward(self, batch_size=None):

        # Sample channel impulse responses
        h, tau = self._cir_sampler( batch_size,
                                    self._num_ofdm_symbols,
                                    self._sampling_frequency)
        #print(h.shape,tau.shape)
        h_freq = cir_to_ofdm_channel(self._frequencies, h, tau,
                                     self._normalize_channel)

        return h_freq


class AWGN(torch.nn.Module):

    def __init__(self):
        super(AWGN,self).__init__()

    def forward(self, x, no):

        # Create tensors of real-valued Gaussian noise for each complex dim.
        noise = complex_normal(x.shape, dtype=x.dtype).to(x.device)

        # Add extra dimensions for broadcasting
        no = expand_to_rank(no, x.ndim, axis=-1)
        # Apply variance scaling
        no = torch.tensor(no, dtype=torch.float32)

        noise = torch.sqrt(no)*noise

        # Add noise to input
        y = x + noise

        return y
class OFDMChannel(torch.nn.Module):
    def __init__(self):
        super(OFDMChannel,self).__init__()
        self._awgn = AWGN()
    def forward(self, x, h_freq, no):
        # Apply the channel response
        x = expand_to_rank(x, h_freq.ndim, axis=1)
        y = torch.sum(torch.sum(h_freq*x, axis=4), axis=3)
        #y = torch.sum(torch.sum( x, axis=4), axis=3)
        # Add AWGN if requested
        y = self._awgn(y, no)
        c = torch.mean(torch.square(torch.abs(y)))
        c = torch.sqrt(c)
        #print(c)
        #y = torch.div(y, c)
        return y,c

class OFDMmodular(torch.nn.Module):
    def __init__(self,cyclic_prefix_length=0):
        super(OFDMmodular,self).__init__()
        self._cyclic_prefix_length=cyclic_prefix_length
    def forward(self,inputs):
        x = torch.fft.ifft(inputs,-1)
        cp = x[..., inputs.shape[-1] - self._cyclic_prefix_length:]
        x=torch.cat([cp, x], -1)
        return x

class OFDMdemodular(torch.nn.Module):
    def __init__(self, fft_size, l_min,cyclic_prefix_length=0):
        super(OFDMdemodular,self).__init__()
        self.fft_size = int(fft_size)
        self.l_min = int(l_min)
        self.cyclic_prefix_length = int(cyclic_prefix_length)

        tmp = -2 * PI * torch.tensor(self.l_min, dtype=torch.float32)\
        / torch.tensor(self.fft_size, dtype=torch.float32)\
        *torch.arange(self.fft_size, dtype=torch.float32)
        self._phase_compensation = torch.exp(torch.complex64(0,tmp))
        self._rest = input_shape[-1] % (self.fft_size + self.cyclic_prefix_length)
        self._num_ofdm_symbols = (input_shape[-1] - self._rest) // (self.fft_size + self.cyclic_prefix_length)
    def forward(self,inputs):
        inputs = inputs if self._rest == 0 else inputs[..., :-self._rest]
        new_shape = torch.cat([torch.tensor(inputs.shape[:-1]), torch.tensor([self._num_ofdm_symbols]),
                               torch.tensor([self.fft_size + self.cyclic_prefix_length])], dim=0)
        x = inputs.reshape(new_shape)
        x = x[..., self.cyclic_prefix_length:]
        x=torch.fft.fft(x,-1)
        rot = torch.tensor(self._phase_compensation, x.dtype)
        rot = expand_to_rank(rot, torch.ndim(x), 0)
        x = x * rot
        return x