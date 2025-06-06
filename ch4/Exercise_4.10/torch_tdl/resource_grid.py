#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition and functions related to the resource grid"""


import numpy as np


from opencood.models.torch_tdl.pilot_pattern import PilotPattern, EmptyPilotPattern, KroneckerPilotPattern # pylint: disable=line-too-long
from opencood.models.torch_tdl.channel_utils_torch import flatten_last_dims
import matplotlib.pyplot as plt
from matplotlib import colors
import torch

class ResourceGrid():
    # pylint: disable=line-too-long
    def __init__(self,
                 num_ofdm_symbols,
                 fft_size,
                 subcarrier_spacing,
                 num_tx=1,
                 num_streams_per_tx=1,
                 cyclic_prefix_length=140,
                 num_guard_carriers=(0,0),
                 dc_null=False,
                 pilot_pattern=None,
                 pilot_ofdm_symbol_indices=None,
                 dtype=torch.complex64):
        super().__init__()
        self._dtype = dtype
        self._num_ofdm_symbols = num_ofdm_symbols
        self._fft_size = fft_size
        self._subcarrier_spacing = subcarrier_spacing
        self._cyclic_prefix_length = int(cyclic_prefix_length)
        self._num_tx = num_tx
        self._num_streams_per_tx = num_streams_per_tx
        self._num_guard_carriers = np.array(num_guard_carriers)
        self._dc_null = dc_null
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self.pilot_pattern = pilot_pattern


    @property
    def cyclic_prefix_length(self):
        """Length of the cyclic prefix."""
        return self._cyclic_prefix_length

    @property
    def num_tx(self):
        """Number of transmitters."""
        return self._num_tx

    @property
    def num_streams_per_tx(self):
        """Number of streams  per transmitter."""
        return self._num_streams_per_tx

    @property
    def num_ofdm_symbols(self):
        """The number of OFDM symbols of the resource grid."""
        return self._num_ofdm_symbols

    @property
    def num_resource_elements(self):
        """Number of resource elements."""
        return self._fft_size*self._num_ofdm_symbols

    @property
    def num_effective_subcarriers(self):
        """Number of subcarriers used for data and pilot transmissions."""
        n = self._fft_size - self._dc_null - np.sum(self._num_guard_carriers)
        return n

    @property
    def effective_subcarrier_ind(self):
        """Returns the indices of the effective subcarriers."""
        num_gc = self._num_guard_carriers
        sc_ind = range(num_gc[0], self.fft_size-num_gc[1])
        if self.dc_null:
            sc_ind = np.delete(sc_ind, self.dc_ind-num_gc[0])
        return sc_ind

    @property
    def num_data_symbols(self):
        """Number of resource elements used for data transmissions."""
        n = self.num_effective_subcarriers * self._num_ofdm_symbols - \
               self.num_pilot_symbols
        return n

    @property
    def num_pilot_symbols(self):
        """Number of resource elements used for pilot symbols."""
        return self.pilot_pattern.num_pilot_symbols

    @property
    def num_zero_symbols(self):
        """Number of empty resource elements."""
        n = (self._fft_size-self.num_effective_subcarriers) * \
               self._num_ofdm_symbols
        return n

    @property
    def num_guard_carriers(self):
        """Number of left and right guard carriers."""
        return self._num_guard_carriers

    @property
    def dc_ind(self):
        """Index of the DC subcarrier.

        If ``fft_size`` is odd, the index is (``fft_size``-1)/2.
        If ``fft_size`` is even, the index is ``fft_size``/2.
        """
        return int(self._fft_size/2 - (self._fft_size%2==1)/2)

    @property
    def fft_size(self):
        """The FFT size."""
        return self._fft_size

    @property
    def subcarrier_spacing(self):
        """The subcarrier spacing [Hz]."""
        return self._subcarrier_spacing

    @property
    def ofdm_symbol_duration(self):
        """Duration of an OFDM symbol with cyclic prefix [s]."""
        return (1. + self.cyclic_prefix_length/self.fft_size) \
                / self.subcarrier_spacing

    @property
    def bandwidth(self):
        """The occupied bandwidth [Hz]: ``fft_size*subcarrier_spacing``."""
        return self.fft_size*self.subcarrier_spacing

    @property
    def num_time_samples(self):
        """The number of time-domain samples occupied by the resource grid."""
        return (self.fft_size + self.cyclic_prefix_length) \
                * self._num_ofdm_symbols

    @property
    def dc_null(self):
        """Indicates if the DC carriers is nulled or not."""
        return self._dc_null

    @property
    def pilot_pattern(self):
        """The used PilotPattern."""
        return self._pilot_pattern

    @pilot_pattern.setter
    def pilot_pattern(self, value):
        if value is None:
            value = EmptyPilotPattern(self._num_tx,
                                      self._num_streams_per_tx,
                                      self._num_ofdm_symbols,
                                      self.num_effective_subcarriers,
                                      dtype=self._dtype)
        elif isinstance(value, PilotPattern):
            pass
        elif isinstance(value, str):
            assert value in ["kronecker", "empty"],\
                "Unknown pilot pattern"
            if value=="empty":
                value = EmptyPilotPattern(self._num_tx,
                                      self._num_streams_per_tx,
                                      self._num_ofdm_symbols,
                                      self.num_effective_subcarriers,
                                      dtype=self._dtype)
            elif value=="kronecker":
                assert self._pilot_ofdm_symbol_indices is not None,\
                    "You must provide pilot_ofdm_symbol_indices."
                value = KroneckerPilotPattern(self,
                        self._pilot_ofdm_symbol_indices, dtype=self._dtype)
        else:
            raise ValueError("Unsupported pilot_pattern")
        self._pilot_pattern = value
    def build_type_grid(self):
        """Returns a tensor indicating the type of each resource element.

        Resource elements can be one of

        - 0 : Data symbol
        - 1 : Pilot symbol
        - 2 : Guard carrier symbol
        - 3 : DC carrier symbol

        Output
        ------
        : [num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.int32
            Tensor indicating for each transmitter and stream the type of
            the resource elements of the corresponding resource grid.
            The type can be one of [0,1,2,3] as explained above.
        """
        # shape = [self._num_tx, self._num_streams_per_tx, self._num_ofdm_symbols]
        # gc_l = 2*torch.ones(shape+[self._num_guard_carriers[0]])
        # gc_r = 2*torch.ones(shape+[self._num_guard_carriers[1]])
        # dc   = 3*torch.ones(shape + [tf.cast(self._dc_null, tf.int32)])
        # mask = self.pilot_pattern.mask
        # split_ind = self.dc_ind-self._num_guard_carriers[0]
        # rg_type = torch.cat([gc_l,                 # Left Guards
        #                      mask[...,:split_ind], # Data & pilots
        #                      dc,                   # DC
        #                      mask[...,split_ind:], # Data & pilots
        #                      gc_r], -1)            # Right guards
        rg_type=self.pilot_pattern.mask
        return rg_type

    def show(self, tx_ind=0, tx_stream_ind=0):
        """Visualizes the resource grid for a specific transmitter and stream.

        Input
        -----
        tx_ind : int
            Indicates the transmitter index.

        tx_stream_ind : int
            Indicates the index of the stream.

        Output
        ------
        : `matplotlib.figure`
            A handle to a matplot figure object.
        """
        fig = plt.figure()
        data = self.build_type_grid()[tx_ind, tx_stream_ind]
        cmap = colors.ListedColormap([[60/256,8/256,72/256],
                              [45/256,91/256,128/256],
                              [45/256,172/256,111/256],
                              [250/256,228/256,62/256]])
        bounds=[0,1,2,3,4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(np.transpose(data), interpolation="nearest",
                         origin="lower", cmap=cmap, norm=norm,
                         aspect="auto")
        cbar = plt.colorbar(img, ticks=[0.5, 1.5, 2.5,3.5],
                            orientation="vertical", shrink=0.8)
        cbar.set_ticklabels(["Data", "Pilot", "Guard carrier", "DC carrier"])
        plt.title("OFDM Resource Grid")
        plt.ylabel("Subcarrier Index")
        plt.xlabel("OFDM Symbol")
        plt.xticks(range(0, data.shape[0]))

        return fig


class ResourceGridMapper(torch.nn.Module):
    # pylint: disable=line-too-long
    r"""
    Input
    -----
    : [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
        The modulated data symbols to be mapped onto the resource grid.

    Output
    ------
    : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        The full OFDM resource grid in the frequency domain.
    """
    def __init__(self, resource_grid):
        super(ResourceGridMapper,self).__init__()
        self._resource_grid = resource_grid
        self._rg_type = self._resource_grid.build_type_grid()
        rg_type_flatten=torch.flatten(self._rg_type)
        pilot_ind= torch.where(rg_type_flatten==1)
        self._pilot_ind=pilot_ind[0]
        data_ind = torch.where(rg_type_flatten == 0)
        self._data_ind= data_ind[0]

    def forward(self, inputs):
        # Map pilots on empty resource grid
        shape=self._rg_type.shape
        #print(shape)
        pilots = flatten_last_dims(self._resource_grid.pilot_pattern.pilots, 3)
        data=flatten_last_dims(inputs, 3)
        pilots =torch.unsqueeze(pilots,0).to(inputs.device)
        pilot_ind=torch.unsqueeze(self._pilot_ind,0).to(inputs.device)

        zero_fill=torch.zeros([1,self._rg_type.shape[2]*self._rg_type.shape[3]],dtype=torch.complex64).to(inputs.device)

        template = torch.scatter(zero_fill,1,pilot_ind,pilots)

        # Broadcast the resource grid template to batch_size
        batch_size = inputs.shape[0]
        template=template.repeat(batch_size,1)
        #print(template.shape)
        data_ind = torch.unsqueeze(self._data_ind, 0).to(inputs.device)
        data_ind = data_ind.repeat(batch_size,1)
        # Flatten the inputs and put batch_dim last for scatter update
        #print(template.shape,data_ind.shape,data.shape)
        template_data=torch.scatter(template,1,data_ind,data)
        rg=template_data.reshape(batch_size,shape[0],shape[1],shape[2],shape[3])

        return rg