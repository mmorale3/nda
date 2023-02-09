// Copyright (c) 2018-2021 Simons Foundation
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
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once

#include <mpi/mpi.hpp>
#include <iostream>
#if defined(NDA_HAVE_CUDA)
#include <cuda_runtime.h>

namespace nda::mem {

inline void device_check(cudaError_t sucess, std::string message = "")
{
  if (sucess != cudaSuccess) {
    std::cerr <<"Cuda runtime error: " <<std::to_string(sucess) <<"\n" 
	      <<" message: " <<message <<"\n"
              <<" cudaGetErrorName: " << std::string(cudaGetErrorName(sucess)) <<"\n"
              <<" cudaGetErrorString: " << std::string(cudaGetErrorString(sucess)) <<"\n";
    mpi::communicator{}.abort(31);
  }
  sucess = cudaGetLastError();
  if (sucess != cudaSuccess) {
    std::cerr <<"Cuda runtime error: " <<std::to_string(sucess) <<"\n"
              <<" message: cudaGetLastError  within device_check "  <<"\n"
              <<" cudaGetErrorName: " << std::string(cudaGetErrorName(sucess)) <<"\n"
              <<" cudaGetErrorString: " << std::string(cudaGetErrorString(sucess)) <<"\n";
    mpi::communicator{}.abort(31);
  }
  sucess = cudaDeviceSynchronize();
  if (sucess != cudaSuccess) {
    std::cerr <<"Cuda runtime error: " <<std::to_string(sucess) <<"\n"
              <<" message: cudaDeviceSynchronize within device_check " <<"\n"
              <<" cudaGetErrorName: " << std::string(cudaGetErrorName(sucess)) <<"\n"
              <<" cudaGetErrorString: " << std::string(cudaGetErrorString(sucess)) <<"\n";
    mpi::communicator{}.abort(31);
  }
}

inline void device_error_check(std::string message) {
    device_check(cudaGetLastError(),message);
    device_check(cudaDeviceSynchronize(), message);
    device_check(cudaGetLastError(),message);
} 

} // nda::mem

#endif

