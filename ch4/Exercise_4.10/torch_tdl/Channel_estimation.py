import torch
from opencood.models.torch_tdl.channel_utils_torch import flatten_last_dims,expand_to_rank,insert_dims
import numpy as np
class ChannelEstimator(torch.nn.Module):
    def __init__(self, resource_grid):
        super(ChannelEstimator, self).__init__()
        self._pilot_pattern = resource_grid.pilot_pattern
        self._interpol =LinearInterpolator_simple(resource_grid)
        num_pilot_symbols = self._pilot_pattern.num_pilot_symbols
        #print(num_pilot_symbols)
        mask = flatten_last_dims(self._pilot_pattern.mask)
        _, pilot_ind= torch.sort(mask, dim=-1, descending=True)
        self._pilot_ind = pilot_ind[..., :num_pilot_symbols]


    def forward(self,y,no):
        y_eff_flat = flatten_last_dims(y)
        #print(y_eff_flat.shape)
        pilot_ind =torch.flatten(self._pilot_ind,start_dim=0,end_dim=-1).to(y.device)

        #y_pilots = torch.index_select(y_eff_flat, axis=-1,index=pilot_ind)
        y_pilots=torch.stack([y[...,2,:],y[...,11,:]],dim=-2)
        #print(y_pilots.shape)
        y_pilots= flatten_last_dims(y_pilots)
        y_pilots=insert_dims(y_pilots,2,-2)
        #print(y_pilots.shape)
        h_hat, err_var = MMSEChannelEstimator(self._pilot_pattern,y_pilots, no)
        #print(self._pilot_pattern.pilots.shape)
        #print(h_hat.shape,err_var.shape)
        h_hat, err_var = self._interpol([h_hat, err_var],pilot_ind)
        err_var = torch.max(err_var, torch.tensor(0))
        return h_hat, err_var

def  LSChannelEstimator(pilot_pattern,y_pilots,no):
    pilot=pilot_pattern.pilots
    pilot=pilot.to(y_pilots.device)
    h_ls = torch.div(y_pilots, pilot)
    #print(y_pilots.shape, pilot.shape)
    #print(h_ls.shape)
    assert torch.isnan(h_ls).sum() == 0
    #print(h_ls.shape)
    no = expand_to_rank(no, h_ls.ndim, -1)
    pilots = expand_to_rank(pilot, h_ls.ndim, 0)
    err_var = torch.div(no, torch.abs(pilots) ** 2)
    return h_ls, err_var

def MMSEChannelEstimator(pilot_pattern,y_pilots,no):
    pilot = pilot_pattern.pilots
    pilot = pilot.to(y_pilots.device)
    #print(y_pilots.shape)
    h_ls = torch.div(y_pilots.squeeze(), pilot.squeeze())
    pilots = expand_to_rank(pilot, h_ls.ndim, 0)
    err_var = torch.div(no, torch.abs(pilots) ** 2)
    size=y_pilots.shape[0]
    h_ls=h_ls.reshape(y_pilots.shape[0]*2,-1,1)
    h_ls_conj=torch.conj(h_ls).permute(0,2,1)
    Rhh=torch.matmul(h_ls,h_ls_conj)
    #print(Rhh.shape)
    Rhh=torch.mean(Rhh,0)
    #print(Rhh)
   # print(Rhh.shape)
    w=torch.matmul(Rhh,torch.inverse(Rhh+no*torch.eye(y_pilots.shape[-1]//2).to(no.device)))
    #print(no*torch.eye(64).to(no.device))
   # print(w.shape)
    h_mmse=torch.matmul(w,h_ls)
    h_mmse=h_mmse.reshape(size,-1)
   # print(h_mmse.shape)
    return h_mmse,err_var

class OnepilotInterpolator(torch.nn.Module):
    def __init__(self,resource_grid):
        super(OnepilotInterpolator, self).__init__()
        self._pilot_pattern = resource_grid.pilot_pattern

    def interpolar(self,input,pilot_ind):
        mask_shape = self._pilot_pattern.mask.shape
        h = input.reshape(input.shape[0], 2, mask_shape[-1])
        h1 = h[:, 0, :].reshape(input.shape[0], 1, mask_shape[-1])
        h_total = h1.repeat(1, 14, 1)

        return h_total

    def forward(self,input,pilot_ind):
        h, err_var = input
        h = h.to(torch.complex64)
        h_total=self.interpolar(h,pilot_ind)
        err_var_total = self.interpolar(err_var, pilot_ind)
        return h_total, err_var_total

class LinearInterpolator_simple(torch.nn.Module):
    def __init__(self, resource_grid):
        super(LinearInterpolator_simple, self).__init__()
        self._pilot_pattern = resource_grid.pilot_pattern

    def interpolator(self,input,pilot_ind):
        mask_shape = self._pilot_pattern.mask.shape
        h = input.reshape(input.shape[0], 2, mask_shape[-1])
        h1=h[:,0,:]
        h2=h[:,1,:]
        h_sort=torch.arange(0,14).to(input.device)
        h_sort=torch.unsqueeze(h_sort,-1)
        h_sort = torch.unsqueeze(h_sort, 0)
        h_sort = h_sort.repeat(input.shape[0],1,mask_shape[-1])
        slope= torch.unsqueeze((h2-h1),1)
        h1= torch.unsqueeze(h1,1)
        h_template= (h_sort-2)/9*slope+h1


        h_template = h_template.reshape(input.shape[0], 1, 1, 1, 1, mask_shape[-2], mask_shape[-1])

        return h_template

    def forward(self,inputs,pilot_ind):
        h,err_var=inputs
        h=h.to(torch.complex64)
        h_total=self.interpolator(h,pilot_ind)
        err_var_total=self.interpolator(err_var,pilot_ind)
        return h_total,err_var_total

class MMSE_equalization(torch.nn.Module):
    def __init__(self):
        super(MMSE_equalization, self).__init__()

    def forward(self,H_est,Y,noise,H_real):
        H_real=torch.squeeze(H_real)
        H_est=torch.squeeze(H_est)
        Y = torch.squeeze(Y)
        #print(H_real,H_est)
        #H_est=H_real
        #print("H_est",H_est.shape)
        #print("Y", Y.shape)
        no =  torch.conj(H_est)*Y
        de =  torch.abs(H_est)**2 + noise
        # if c>1.5:
        #     print(c)
        #print(de)
        #print(torch.nonzero(torch.div(no,de)-10).shape[0])
        return torch.div(no,de+0.1)

