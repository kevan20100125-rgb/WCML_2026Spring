from resource_grid import ResourceGrid,ResourceGridMapper
from OFDM import GenerateOFDMChannel
from Channel_estimation import ChannelEstimator
from OFDM_TDL import OFDM_TDL
import torch


# real=torch.stack([torch.zeros([1,1,1,12*64]),torch.zeros([1,1,1,12*64])],0).reshape(2,1,1,12*64)
# input =torch.complex(real,torch.zeros([2,1,1,12*64]))
input =torch.zeros([256,42])
# no = torch.zeros([2,1,1])
#
# rg = ResourceGrid(num_ofdm_symbols=14,
#                  fft_size=64,
#                  subcarrier_spacing = 30e3,
#                  num_tx=1,
#                   num_streams_per_tx=1,
#                   pilot_pattern = "kronecker",
#                   pilot_ofdm_symbol_indices = [2, 11])
# rg.show()


# gen=GenerateOFDMChannel(resource_grid=rg)
# h_f=gen(batch_size=2)
# print(h_f.shape)

OFDM_TDL_=OFDM_TDL()
for snr in range(20):
    data,H_MMSE,last_H,H_freq_pow=OFDM_TDL_(input,SNR=snr,ACK_times=0,H_last=None)

    print(snr,H_MMSE)