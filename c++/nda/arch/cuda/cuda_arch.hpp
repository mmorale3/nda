// Copyright (c) 2018-2022 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Miguel Morales

#ifndef ARCH_CUDA_HPP
#define ARCH_CUDA_HPP

#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <iostream>

#include "../../exceptions.hpp"
#include "../../traits.hpp"

#include <cuda_runtime.h>

namespace nda::arch::device
{

  inline void cuda_check(cudaError_t sucess, std::string message = "")
  {
    if (cudaSuccess != sucess) {
      NDA_RUNTIME_ERROR << message <<"\n"
                        << " cudaGetErrorName: " << cudaGetErrorName(sucess) << "\n"
                        << " cudaGetErrorString: " << cudaGetErrorString(sucess) << "\n"; 
    }
  }

  enum MEMCOPYKIND
  {
    memcopyH2H     = cudaMemcpyHostToHost,
    memcopyH2D     = cudaMemcpyHostToDevice,
    memcopyD2H     = cudaMemcpyDeviceToHost,
    memcopyD2D     = cudaMemcpyDeviceToDevice,
    memcopyDefault = cudaMemcpyDefault
  };

  namespace detail {

    inline cudaMemcpyKind tocudaMemcpyKind(MEMCOPYKIND v)
    {
      switch (v)
      {
        case memcopyH2H: 
          return cudaMemcpyHostToHost;
        case memcopyH2D: 
          return cudaMemcpyHostToDevice;
        case memcopyD2H: 
          return cudaMemcpyDeviceToHost;
        case memcopyD2D: 
          return cudaMemcpyDeviceToDevice;
        case memcopyDefault: 
          return cudaMemcpyDefault;
      }
      return cudaMemcpyDefault;
    }

  }

  inline void memset(void* devPtr, int value, size_t count)
  { 
    cuda_check( cudaMemset(devPtr, value, count), "cudaMemset failed:" );
  }

  inline void memset2D(void* devPtr, size_t p, int value, size_t w, size_t h)
  {
    cuda_check( cudaMemset2D(devPtr, p, value, w, h), "cudaMemset2D failed:" );
  }

  inline void memcopy(void* dst, const void* src, size_t count, MEMCOPYKIND kind)
  {
    cuda_check( cudaMemcpy(dst, src, count, detail::tocudaMemcpyKind(kind)), "cudaMemcpy failed:" );
  }

  inline void memcopy2D(void* dst, size_t dpitch, const void* src, size_t spitch, 
			size_t width, size_t height, MEMCOPYKIND kind)
  {
    cuda_check( cudaMemcpy2D(dst, dpitch, src, spitch, width, height, detail::tocudaMemcpyKind(kind)) , 
		"cudaMemcpy2D failed:" );
  }

  inline void malloc(void** devPtr, size_t size)
  {
    cuda_check( cudaMalloc(devPtr, size), "cudaMalloc failed:" );
  }

  inline void free(void* p)
  {
    cuda_check( cudaFree(p), "cudaFree failed:" );
  }

  template< class T, class Size >
  requires(nda::is_scalar_or_convertible_v<T>) 
  void fill_n( T* first, Size count, const T& value )
  {
    if(std::find_if((char const*)(&value), (char const*)(&value) + sizeof(T), [](char c){return c!=0;}) == (char const*)(&value) + sizeof(T)){
      memset((void*)first,0,count*sizeof(T));
    } else {
      // MAM: temporary, use kernel/thrust/foreach/... when available
      int v=0;
      uint8_t const* ui = reinterpret_cast<uint8_t const*>(&value);
      uint8_t *fn = reinterpret_cast<uint8_t*>(first);
      for(int n=0; n<sizeof(T); ++n) {
        v=0; // just in case
        v = *(ui+n);
        memset2D((void*)(fn+n), sizeof(T), v, 1, count);
      }   
    }
  }

  template< class T>
  void fill( T* beg, T* end, const T& value )
  {
    fill_n(beg,std::distance(beg,end),value);
  }

  template< class T, class Size >
  requires(nda::is_scalar_or_convertible_v<T>)
  void fill2D_n( T* first, Size pitch, Size width, Size height, const T& value )
  {
    if(std::find_if((char const*)(&value), (char const*)(&value) + sizeof(T), [](char c){return c!=0;}) == (char const*)(&value) + sizeof(T)){
      memset2D((void*)first,pitch*sizeof(T),0,width*sizeof(T),height);
    } else {
      // MAM: temporary, use kernel/thrust/foreach/... when available
      // as a temporary version, can also loop over rows...
      nda::array<T,1> buf(width*height);
      buf()=value;
      memcopy2D((void*)first, pitch*sizeof(T), (void*) buf.data(), width*sizeof(T), 
                width*sizeof(T), height, memcopyH2D);
    }
  }

} // nda::arch::device

#endif
