import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray as ga
from pycuda import characterize

from chroma.gpu.tools import get_cu_module, cuda_options, \
    chunk_iterator, to_uint3, to_float3, GPUFuncs, mapped_empty, Mapped

from chroma.bvh.bvh import WorldCoords

def round_up_to_multiple(x, multiple):
    remainder = x % multiple
    if remainder == 0:
        return x
    else:
        return x + multiple - remainder

def create_leaf_nodes(mesh, morton_bits=16, round_to_multiple=1):
    '''Compute the leaf nodes surrounding a triangle mesh.

      ``mesh``: chroma.geometry.Mesh
        Triangles to box
      ``morton_bits``: int
        Number of bits to use per dimension when computing Morton code.
      ``round_to_multiple``: int
        Round the number of nodes created up to multiple of this number
        Extra nodes will be all zero.
        
    Returns (world_coords, nodes, morton_codes), where
      ``world_coords``: chroma.bvh.WorldCoords
        Defines the fixed point coordinate system
      ``nodes``: ndarray(shape=len(mesh.triangles), dtype=uint4)
        List of leaf nodes.  Child IDs will be set to triangle offsets.
      ``morton_codes``: ndarray(shape=len(mesh.triangles), dtype=np.uint64)
        Morton codes for each triangle, using ``morton_bits`` per axis.
        Must be <= 16 bits.
    '''
    # Load GPU functions
    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
                               include_source_directory=True)
    bvh_funcs = GPUFuncs(bvh_module)

    # compute world coordinates
    world_origin = mesh.vertices.min(axis=0)
    world_scale = np.max((mesh.vertices.max(axis=0) - world_origin)) \
        / (2**16 - 2)
    world_coords = WorldCoords(world_origin=world_origin, 
                               world_scale=world_scale)

    # Put triangles and vertices in mapped host memory
    triangles = mapped_empty(shape=len(mesh.triangles), dtype=ga.vec.uint3,
                             write_combined=True)
    triangles[:] = to_uint3(mesh.triangles)
    vertices = mapped_empty(shape=len(mesh.vertices), dtype=ga.vec.float3,
                            write_combined=True)
    vertices[:] = to_float3(mesh.vertices)
    
    # Call GPU to compute nodes
    nodes = ga.zeros(shape=round_up_to_multiple(len(triangles), 
                                                round_to_multiple),
                     dtype=ga.vec.uint4)
    morton_codes = ga.empty(shape=len(triangles), dtype=np.uint64)

    # Convert world coords to GPU-friendly types
    world_origin = ga.vec.make_float3(*world_origin)
    world_scale = np.float32(world_scale)

    nthreads_per_block = 256
    for first_index, elements_this_iter, nblocks_this_iter in \
            chunk_iterator(len(triangles), nthreads_per_block, 
                           max_blocks=30000):
        bvh_funcs.make_leaves(np.uint32(first_index),
                              np.uint32(elements_this_iter),
                              Mapped(triangles), Mapped(vertices),
                              world_origin, world_scale,
                              nodes, morton_codes,
                              block=(nthreads_per_block,1,1),
                              grid=(nblocks_this_iter,1))

    morton_codes_host = morton_codes.get() >> (16 - morton_bits)
    return world_coords, nodes.get(), morton_codes_host


def merge_nodes(nodes, degree):
    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
                               include_source_directory=True)
    bvh_funcs = GPUFuncs(bvh_module)
    
    nparent = len(nodes) / degree
    if nparent == 1:
        nparent_pad = nparent
    else:
        nparent_pad = round_up_to_multiple(nparent, degree)
    parent_nodes = ga.zeros(shape=nparent_pad, dtype=ga.vec.uint4)

    nthreads_per_block = 256
    for first_index, elements_this_iter, nblocks_this_iter in \
            chunk_iterator(nparent, nthreads_per_block, max_blocks=10000):
        bvh_funcs.make_parents(np.uint32(first_index),
                               np.uint32(elements_this_iter),
                               np.uint32(degree),
                               parent_nodes,
                               cuda.In(nodes),
                               np.uint32(0),
                               block=(nthreads_per_block,1,1),
                               grid=(nblocks_this_iter,1))

    return parent_nodes.get()

def concatenate_layers(layers):
    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
                               include_source_directory=True)
    bvh_funcs = GPUFuncs(bvh_module)
    # Put 0 at beginning of list
    layer_bounds = np.insert(np.cumsum(map(len, layers)), 0, 0)
    nodes = ga.empty(shape=int(layer_bounds[-1]), dtype=ga.vec.uint4)
    nthreads_per_block = 256

    for layer_start, layer_end, layer in zip(layer_bounds[:-1],
                                             layer_bounds[1:],
                                             layers):
        if layer_end == layer_bounds[-1]:
            # leaf nodes need no offset
            child_offset = 0
        else:
            child_offset = layer_end

        for first_index, elements_this_iter, nblocks_this_iter in \
                chunk_iterator(layer_end-layer_start, nthreads_per_block,
                               max_blocks=10000):
            bvh_funcs.copy_and_offset(np.uint32(first_index),
                                      np.uint32(elements_this_iter),
                                      np.uint32(child_offset),
                                      cuda.In(layer),
                                      nodes[layer_start:],
                                      block=(nthreads_per_block,1,1),
                                      grid=(nblocks_this_iter,1))
        
    return nodes.get(), layer_bounds
