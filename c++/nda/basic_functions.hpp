// Copyright (c) 2019-2021 Simons Foundation
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

#include "basic_array.hpp"

namespace nda {

  // --------------------------- zeros ------------------------

  /// Make an array of the given dimensions and zero-initialized values / memory.
  /// Return a scalar for the case of rank zero.
  /// For a more specific array type consider using basic_array<...>::zeros
  template <typename T, std::integral Int, auto Rank>
  auto zeros(std::array<Int, Rank> const &shape) {
    // For Rank == 0 we should return the underlying scalar_t
    if constexpr (Rank == 0)
      return T{};
    else
      return array<T, Rank>::zeros(shape);
  }

  ///
  template <typename T, std::integral... Int>
  auto zeros(Int... i) {
    return zeros<T>(std::array<long, sizeof...(Int)>{i...});
  }

  // --------------------------- ones ------------------------

  /// Make an array of the given dimensions holding 'scalar ones'.
  /// Return a scalar for the case of rank zero.
  /// For a more specific array type consider using basic_array<...>::ones
  template <typename T, std::integral Int, auto Rank>
  auto ones(std::array<Int, Rank> const &shape) requires(nda::is_scalar_v<T>) {
    // For Rank == 0 we should return the underlying scalar_t
    if constexpr (Rank == 0)
      return T{1};
    else {
      return array<T, Rank>::ones(shape);
    }
  }

  ///
  template <typename T, std::integral... Int>
  auto ones(Int... i) {
    return ones<T>(std::array<long, sizeof...(Int)>{i...});
  }

  // --------------------------- rand ------------------------

  /// Create an array the given dimensions and populate it with random
  /// samples from a uniform distribution over [0, 1).
  /// Return a scalar for the case of rank zero.
  /// For a more specific array type consider using basic_array<...>::rand
  template <typename RealType = double, std::integral Int, auto Rank>
  auto rand(std::array<Int, Rank> const &shape) requires(std::is_floating_point_v<RealType>) {
    // For Rank == 0 we should return a scalar
    if constexpr (Rank == 0) {
      auto static gen  = std::mt19937(std::random_device{}());
      auto static dist = std::uniform_real_distribution<>(0.0, 1.0);
      return dist(gen);
    } else {
      return array<RealType, Rank>::rand(shape);
    }
  }

  ///
  template <typename RealType = double, std::integral... Int>
  auto rand(Int... i) {
    return rand<RealType>(std::array<long, sizeof...(Int)>{i...});
  }

  // --------------------------- dim helpers ------------------------

  /// Return the first array dimension
  /// @tparam A Type modeling NdArray
  /// @param a Object
  /// @return The first dimension. Equivalent to a.extent(0) and a.shape()[0]
  template <Array A>
  long first_dim(A const &a) { return a.extent(0); }

  /// Return the second array dimension
  /// @tparam A Type modeling NdArray
  /// @param a Object
  /// @return The second dimension. Equivalent to a.extent(1) and a.shape()[1]
  template <Array A>
  long second_dim(A const &a) { return a.extent(1); }

  // --------------------------- make_regular ------------------------

  /**
   * Return a basic_array if A fullfills the Array concept,
   * else forward the object without midifications
   *
   * @tparam A
   * @param x
   */
  template <typename A>
  auto make_regular(A &&x) {
    using A_t = std::decay_t<A>;
    if constexpr (Array<A>)
      return basic_array<get_value_t<A_t>, get_rank<A_t>, C_layout, get_algebra<A_t>, heap>{std::forward<A>(x)};
    else
      return std::forward<A>(x);
  }

  template <typename A>
  using get_regular_t = decltype(make_regular(std::declval<A>()));

  // --------------------------- resize_or_check_if_view------------------------

  /** 
   * Resize if A is a container, or assert that the view has the right dimension if A is view
   *
   * @tparam A
   * @param a A container or a view
   */
  template <typename A>
  void resize_or_check_if_view(A &a, std::array<long, A::rank> const &sha) requires(is_regular_or_view_v<A>) {
    if (a.shape() == sha) return;
    if constexpr (is_regular_v<A>) {
      a.resize(sha);
    } else {
      NDA_RUNTIME_ERROR << "Size mismatch : view class shape = " << a.shape() << " expected " << sha;
    }
  }

  // --------------- make_const_view------------------------

  /// Make a view const
  template <typename T, int R, typename L, char Algebra, typename CP>
  basic_array_view<T const, R, L, Algebra> make_const_view(basic_array<T, R, L, Algebra, CP> const &a) {
    return {a};
  }

  template <typename T, int R, typename L, char Algebra, typename AP, typename OP>
  basic_array_view<T const, R, L, Algebra, AP, OP> make_const_view(basic_array_view<T, R, L, Algebra, AP, OP> const &a) {
    return {a};
  }

  // --------------- make_array_view------------------------

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  array_view<T, R> make_array_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
    return {a};
  }

  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  array_view<T, R> make_array_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> const &a) {
    return {a};
  }

  // --------------- make_array_const_view------------------------

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  array_const_view<T, R> make_array_const_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
    return {a};
  }

  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  array_const_view<T, R> make_array_const_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> const &a) {
    return {a};
  }

  // --------------- make_matrix_view------------------------

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  matrix_view<T, L> make_matrix_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
    return {a};
  }

  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  matrix_view<T, L> make_matrix_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> const &a) {
    return {a};
  }

  /*  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy>*/
  //matrix_view<T const, L> make_matrix_const_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
  //return {a};
  //}

  //template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  //matrix_view<T const, L> make_matrix_const_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> const &a) {
  //return {a};
  //}

  // --------------- operator == ---------------------

  /// True iif all elements are equal.
  template <Array A, Array B>
  bool operator==(A const &a, B const &b) {
 // FIXME not implemented in clang .. readd when done for better error message
#ifndef __clang__
    static_assert(std::equality_comparable_with<get_value_t<A>, get_value_t<B>>, "A == B is only defined when their element can be compared");
#endif
    if (a.shape() != b.shape()) return false;
    bool r = true;
    nda::for_each(a.shape(), [&](auto &&...x) { r &= (a(x...) == b(x...)); });
    return r;
  }

  /// -- Value-comparison with 1D Contiguous Ranges

  template <ArrayOfRank<1> A, std::ranges::contiguous_range R>
  bool operator==(A const &a, R const &r) {
    return a == basic_array_view{r};
  }

  template <std::ranges::contiguous_range R, ArrayOfRank<1> A>
  bool operator==(R const &r, A const &a) {
    return a == r;
  }

  // ------------------------------- auto_assign --------------------------------------------

  template <Array A, typename F>
  void clef_auto_assign(A &&a, F &&f) {
    nda::for_each(a.shape(), [&a, &f](auto &&...x) {
      if constexpr (clef::is_function<std::decay_t<decltype(f(x...))>>) {
        clef_auto_assign(a(x...), f(x...));
      } else {
        a(x...) = f(x...);
      }
    });
  }

} // namespace nda
