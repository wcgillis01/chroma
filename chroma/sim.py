#!/usr/bin/env python
import time
import os
import numpy as np

from chroma import gpu
from chroma import event
from chroma import itertoolset
from chroma.tools import profile_if_possible

import pycuda.driver as cuda

def pick_seed():
    """Returns a seed for a random number generator selected using
    a mixture of the current time and the current process ID."""
    return int(time.time()) ^ (os.getpid() << 16)

class Simulation(object):
    def __init__(self, detector, seed=None, cuda_device=None, nthreads_per_block=64, max_blocks=1024):
        self.detector = detector

        self.nthreads_per_block = nthreads_per_block
        self.max_blocks = max_blocks

        if seed is None:
            self.seed = pick_seed()
        else:
            self.seed = seed

        # We have two generators to seed: numpy.random, and CURAND.
        # The latter two are done below.
        np.random.seed(self.seed)

        self.context = gpu.create_cuda_context(cuda_device)

        if hasattr(detector, 'num_channels'):
            self.gpu_geometry = gpu.GPUDetector(detector)
            self.gpu_daq = gpu.GPUDaq(self.gpu_geometry)
            self.gpu_pdf = gpu.GPUPDF()
            self.gpu_pdf_kernel = gpu.GPUKernelPDF()
        else:
            self.gpu_geometry = gpu.GPUGeometry(detector)

        self.rng_states = gpu.get_rng_states(self.nthreads_per_block*self.max_blocks, seed=self.seed)

        self.pdf_config = None

    def simulate(self, photons, keep_photons_beg=False, keep_photons_end=False, 
                 keep_hits=True, run_daq=False, max_steps=100):
        ev = event.Event(photons_beg=photons)
        
        gpu_photons = gpu.GPUPhotons(ev.photons_beg)

        gpu_photons.propagate(self.gpu_geometry, self.rng_states,
                              nthreads_per_block=self.nthreads_per_block,
                              max_blocks=self.max_blocks,
                              max_steps=max_steps)

        ev.nphotons = len(ev.photons_beg.pos)

        if not keep_photons_beg:
            ev.photons_beg = None

        if keep_photons_end:
            ev.photons_end = gpu_photons.get()
        
        if hasattr(self.detector, 'num_channels') and keep_hits:
            ev.hits = gpu_photons.get_hits(self.gpu_geometry)
            
        # Skip running DAQ if we don't have one
        # Disabled by default because incredibly special-case
        if hasattr(self, 'gpu_daq') and run_daq:
            self.gpu_daq.begin_acquire()
            self.gpu_daq.acquire(gpu_photons, self.rng_states, nthreads_per_block=self.nthreads_per_block, max_blocks=self.max_blocks)
            gpu_channels = self.gpu_daq.end_acquire()
            ev.channels = gpu_channels.get()

        return ev

    def create_pdf(self, iterable, tbins, trange, qbins, qrange, nreps=1):
        """Returns tuple: 1D array of channel hit counts, 3D array of
        (channel, time, charge) pdfs."""
        first_element, iterable = itertoolset.peek(iterable)

        if isinstance(first_element, event.Event):
            iterable = self.photon_generator.generate_events(iterable)

        pdf_config = (tbins, trange, qbins, qrange)
        if pdf_config != self.pdf_config:
            self.pdf_config = pdf_config
            self.gpu_pdf.setup_pdf(self.detector.num_channels(), tbins, trange,
                                   qbins, qrange)
        else:
            self.gpu_pdf.clear_pdf()

        if nreps > 1:
            iterable = itertoolset.repeating_iterator(iterable, nreps)

        for ev in iterable:
            gpu_photons = gpu.GPUPhotons(ev.photons_beg)
            gpu_photons.propagate(self.gpu_geometry, self.rng_states,
                                  nthreads_per_block=self.nthreads_per_block,
                                  max_blocks=self.max_blocks)
            self.gpu_daq.begin_acquire()
            self.gpu_daq.acquire(gpu_photons, self.rng_states, nthreads_per_block=self.nthreads_per_block, max_blocks=self.max_blocks)
            gpu_channels = self.gpu_daq.end_acquire()
            self.gpu_pdf.add_hits_to_pdf(gpu_channels)
        
        return self.gpu_pdf.get_pdfs()

    @profile_if_possible
    def eval_pdf(self, event_channels, iterable, min_twidth, trange, min_qwidth, qrange, min_bin_content=100, nreps=1, ndaq=1, nscatter=1, time_only=True):
        """Returns tuple: 1D array of channel hit counts, 1D array of PDF
        probability densities."""
        ndaq_per_rep = 64
        ndaq_reps = ndaq // ndaq_per_rep
        gpu_daq = gpu.GPUDaq(self.gpu_geometry, ndaq=ndaq_per_rep)

        self.gpu_pdf.setup_pdf_eval(event_channels.hit,
                                    event_channels.t,
                                    event_channels.q,
                                    min_twidth,
                                    trange,
                                    min_qwidth,
                                    qrange,
                                    min_bin_content=min_bin_content,
                                    time_only=True)

        first_element, iterable = itertoolset.peek(iterable)

        if isinstance(first_element, event.Event):
            iterable = self.photon_generator.generate_events(iterable)
        elif isinstance(first_element, event.Photons):
            iterable = (event.Event(photons_beg=x) for x in iterable)

        for ev in iterable:
            gpu_photons_no_scatter = gpu.GPUPhotons(ev.photons_beg, ncopies=nreps)
            gpu_photons_scatter = gpu.GPUPhotons(ev.photons_beg, ncopies=nreps*nscatter)
            gpu_photons_no_scatter.propagate(self.gpu_geometry, self.rng_states,
                                             nthreads_per_block=self.nthreads_per_block,
                                             max_blocks=self.max_blocks,
                                             use_weights=True,
                                             scatter_first=-1,
                                             max_steps=10)
            gpu_photons_scatter.propagate(self.gpu_geometry, self.rng_states,
                                          nthreads_per_block=self.nthreads_per_block,
                                          max_blocks=self.max_blocks,
                                          use_weights=True,
                                          scatter_first=1,
                                          max_steps=5)
            nphotons = gpu_photons_no_scatter.true_nphotons # same for scatter
            for i in xrange(gpu_photons_no_scatter.ncopies):
                start_photon = i * nphotons
                gpu_photon_no_scatter_slice = gpu_photons_no_scatter.select(event.SURFACE_DETECT,
                                                                            start_photon=start_photon,
                                                                            nphotons=nphotons)
                gpu_photon_scatter_slices = [gpu_photons_scatter.select(event.SURFACE_DETECT,
                                                                        start_photon=(nscatter*i+j)*nphotons,
                                                                        nphotons=nphotons)
                                             for j in xrange(nscatter)]
                
                if len(gpu_photon_no_scatter_slice) == 0:
                    continue

                #weights = gpu_photon_slice.weights.get()
                #print 'weights', weights.min(), weights.max()
                for j in xrange(ndaq_reps):
                    gpu_daq.begin_acquire()
                    gpu_daq.acquire(gpu_photon_no_scatter_slice, self.rng_states, nthreads_per_block=self.nthreads_per_block, max_blocks=self.max_blocks)
                    for scatter_slice in gpu_photon_scatter_slices:
                        gpu_daq.acquire(scatter_slice, self.rng_states, nthreads_per_block=self.nthreads_per_block, max_blocks=self.max_blocks, weight=1.0/nscatter)
                    gpu_channels = gpu_daq.end_acquire()
                    self.gpu_pdf.accumulate_pdf_eval(gpu_channels, nthreads_per_block=ndaq_per_rep)
        
        return self.gpu_pdf.get_pdf_eval()

    def setup_kernel(self, event_channels, bandwidth_iterable,
                         trange, qrange, 
                         nreps=1, ndaq=1, time_only=True, scale_factor=1.0):
        '''Call this before calling eval_pdf_kernel().  Sets up the
        event information and computes an appropriate kernel bandwidth'''
        nchannels = len(event_channels.hit)
        self.gpu_pdf_kernel.setup_moments(nchannels, trange, qrange,
                                          time_only=time_only)
        # Compute bandwidth
        first_element, bandwidth_iterable = itertoolset.peek(bandwidth_iterable)
        if isinstance(first_element, event.Event):
            bandwidth_iterable = \
                self.photon_generator.generate_events(bandwidth_iterable)
        for ev in bandwidth_iterable:
            gpu_photons = gpu.GPUPhotons(ev.photons_beg, ncopies=nreps)
            gpu_photons.propagate(self.gpu_geometry, self.rng_states,
                                  nthreads_per_block=self.nthreads_per_block,
                                  max_blocks=self.max_blocks)
            for gpu_photon_slice in gpu_photons.iterate_copies():
                for idaq in xrange(ndaq):
                    self.gpu_daq.begin_acquire()
                    self.gpu_daq.acquire(gpu_photon_slice, self.rng_states, nthreads_per_block=self.nthreads_per_block, max_blocks=self.max_blocks)
                    gpu_channels = self.gpu_daq.end_acquire()
                    self.gpu_pdf_kernel.accumulate_moments(gpu_channels)
            
        self.gpu_pdf_kernel.compute_bandwidth(event_channels.hit,
                                              event_channels.t,
                                              event_channels.q,
                                              scale_factor=scale_factor)

    def eval_kernel(self, event_channels,
                    kernel_iterable,
                    trange, qrange, 
                    nreps=1, ndaq=1, naverage=1, time_only=True):
        """Returns tuple: 1D array of channel hit counts, 1D array of PDF
        probability densities."""

        self.gpu_pdf_kernel.setup_kernel(event_channels.hit,
                                         event_channels.t,
                                         event_channels.q)
        first_element, kernel_iterable = itertoolset.peek(kernel_iterable)
        if isinstance(first_element, event.Event):
            kernel_iterable = \
                self.photon_generator.generate_events(kernel_iterable)

        # Evaluate likelihood using this bandwidth
        for ev in kernel_iterable:
            gpu_photons = gpu.GPUPhotons(ev.photons_beg, ncopies=nreps)
            gpu_photons.propagate(self.gpu_geometry, self.rng_states,
                                  nthreads_per_block=self.nthreads_per_block,
                                  max_blocks=self.max_blocks)
            for gpu_photon_slice in gpu_photons.iterate_copies():
                for idaq in xrange(ndaq):
                    self.gpu_daq.begin_acquire()
                    self.gpu_daq.acquire(gpu_photon_slice, self.rng_states, nthreads_per_block=self.nthreads_per_block, max_blocks=self.max_blocks)
                    gpu_channels = self.gpu_daq.end_acquire()
                    self.gpu_pdf_kernel.accumulate_kernel(gpu_channels)
        
        return self.gpu_pdf_kernel.get_kernel_eval()

    def __del__(self):
        self.context.pop()
