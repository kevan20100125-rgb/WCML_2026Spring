import torch
from opencood.models.torch_tdl.OFDM import OFDMChannel,GenerateOFDMChannel
from opencood.models.torch_tdl.Channel_estimation import ChannelEstimator,MMSE_equalization
from opencood.models.torch_tdl.resource_grid import ResourceGrid,ResourceGridMapper
from opencood.models.encoder_decoder.encoder_4 import Encoder
from opencood.models.encoder_decoder.decoder_4 import Decoder
class OFDM_TDL(torch.nn.Module):
    def __init__(self):
        super(OFDM_TDL, self).__init__()
        self.rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=2048,
                          subcarrier_spacing=15e3,
                          num_tx=1,
                          num_streams_per_tx=1,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2,11])
        self.fft_size=2048
        self.channel_estimation=ChannelEstimator(self.rg)
        self.resourcegrid_mapper=ResourceGridMapper(self.rg)
        self.OFDMChannel=OFDMChannel()
        self.MMSE_equalization=MMSE_equalization()
        self.GenerateOFDMChannel=GenerateOFDMChannel(self.rg,normalize_channel=True)#这里将channel的能量变为normalize为1
        self.encoder1 = Encoder()
        self.decoder1 = Decoder()
    def forward(self,x,SNR,ACK_times,H_last):
        no = 10**(-SNR/10)
        no = torch.tensor(no).to(x.device)
        assert torch.isnan(x).sum() == 0
        #print(x.shape)
        x=self.encoder1(x)
        #print(x.shape)
        #print('encoder ready')
        x_1=x
        assert torch.isnan(x).sum() == 0

        x= torch.squeeze(x).permute(0,2,1)
        #print(x.shape)
        x= x.reshape(2,12,-1)
        shape_s=x.shape[1]
        dc_l=(self.fft_size-x.shape[2])//2
        zero_padd= torch.zeros(x.shape[0],x.shape[1],dc_l).to(x.device)
        x = torch.concat([x,zero_padd],dim=2)
        x = torch.concat([zero_padd,x],dim=2)
        #print(x.shape)
        x_complex=torch.complex(x[0,:,:],x[1,:,:])
        #print(x_complex.shape)
        x=x_complex.reshape(-1,1,1,12*self.fft_size)
        #print(x.shape)
        #print(ACK_times)
        if ACK_times==0:
            #print("generate")
            H_freq_100=self.GenerateOFDMChannel(x.shape[0]).to(x.device)
        else:
            H_freq_100=H_last
        last_H=H_freq_100
        H_freq=H_freq_100[:,:,:,:,:,2*ACK_times*14:(2*ACK_times+1)*14,:]
        #h_freq_pow = H_freq.reshape(2048, 14)
        H_freq_pow = torch.sqrt(torch.mean(torch.square(torch.abs(H_freq)), dim=(2, 4, 5, 6), keepdim=True))
        #H_freq=torch.div(H_freq,H_freq_pow)
        #print(H_freq_pow)

        #H_freq=torch.ones_like(H_freq)
        #print(H_freq.shape)

        data_x= self.resourcegrid_mapper(x)
        data_y,factor=self.OFDMChannel(data_x,H_freq,no)
        #H_freq=torch.div(H_freq,factor)
        #no=torch.div(no,factor)
        H_hat,err_var=self.channel_estimation(data_y,no)
        #print(H_hat)
        data = self.MMSE_equalization(H_hat,data_y,no,H_freq)
        #c = torch.mean(torch.square(torch.abs(data)))
        #c = torch.sqrt(c)
        #print(c)
        #data = torch.div(data, c)
        #data=data_y
        #print(data.shape)


        data =torch.concat([data [0:2,:],data [3:11,:],data [12:14,:]],dim=0)
        data = data[:,dc_l:dc_l+14*64]
        data=data.reshape(-1,256).permute(1,0)
        #print(data.shape)
        #data=data[:,0:shape_s] 
        #print(data.shape)
        data=torch.stack([data.real,data.imag]).unsqueeze(0)
        data =torch.tensor(data,dtype=torch.float32)
        #data_1=data
        assert torch.isnan(data).sum() == 0
        data=self.decoder1(data)
        assert torch.isnan(data).sum() == 0
        # if torch.mean(torch.abs(data_1-x_1)**2)>0.1:
        #     print(torch.mean(torch.abs(data_1-x_1)**2))
        #print(data)
        H_freq=torch.squeeze(H_freq)
        H_hat=torch.squeeze(H_hat)
        H_MMSE=torch.mean(torch.abs(H_freq-H_hat)**2)

        data = self.decoder1(x_1)
        return data,H_MMSE,last_H,H_freq_pow
