#pragma once

namespace nda {

 /* // --------------------------- is_instantiation_of ------------------------*/

  /**
   * is_instantiation_of_v
   * Checks that X is a T<....>
   */
  //template <typename T, template <typename..., auto ...> class TMPLT>
  //inline constexpr bool is_instantiation_of_v = false;

  //template <template <typename..., auto ...> class TMPLT, typename... U, auto ... X>
  //inline constexpr bool is_instantiation_of_v<TMPLT<U..., X...>, TMPLT> = true;

  // --------------------------- is_complex ------------------------

  template <typename T>
  struct _is_complex : std::false_type {};

  template <typename T>
  struct _is_complex<std::complex<T>> : std::true_type {};

  template <typename T>
  inline constexpr bool is_complex_v = _is_complex<std::decay_t<T>>::value;

  // --------------------------- is_scalar ------------------------

  template <typename S>
  inline constexpr bool is_scalar_v = std::is_arithmetic_v<S> or nda::is_complex_v<S>; // painful without the decay in later code

  template <typename S>
  inline constexpr bool is_scalar_or_convertible_v = is_scalar_v<S> or std::is_constructible_v<std::complex<double>, S>;

  template <typename S, typename A>
  inline constexpr bool is_scalar_for_v = (is_scalar_v<typename A::value_t> ? is_scalar_or_convertible_v<S> : std::is_same_v<S, typename A::value_t>);

  // ---------------------------  is_regular_or_view_v------------------------
 
  // Impl. trait to match the containers in requires. Match all containers (array, matrix, view)
  template<typename A> inline constexpr bool is_regular_or_view_v = false;

  // --------------------------- Ndarray concept------------------------

  /// A trait to mark classes modeling the Ndarray concept
  template <typename T>
  inline constexpr bool is_ndarray_v = false;

  // --------------------------- get_rank ------------------------

  /// A trait to get the rank of an object with ndarray concept
  template <typename A>
  constexpr int get_rank = std::tuple_size_v<std::decay_t<decltype(std::declval<A const>().shape())>>;

  // --------------------------- get_value_t ------------------------

  // FIXME C++20 lambda
  template <size_t... Is, typename A>
  auto _get_value_t_impl(std::index_sequence<Is...>, A a) {
    return a((0 * Is)...); // repeat 0 sizeof...(Is) times
  }

  /// A trait to get the return_t of the (long, ... long) for an object with ndarray concept
  template <typename A>
  using get_value_t = decltype(_get_value_t_impl(std::make_index_sequence<get_rank<A>>(), std::declval<A const>()));

  // --------------------------- Algebra ------------------------

  /// A trait to mark a class for its algebra : 'N' = None, 'A' = array, 'M' = matrix, 'V' = vector
  template <typename A>
  inline constexpr char get_algebra = 'N';

  // --------------------------- make_regular ------------------------
  // FIXME MOVE THIS : A function, not a traits
  // general make_regular
  template <typename A>
  typename A::regular_t make_regular(A &&x) REQUIRES(is_ndarray_v<A>) {
    return std::forward<A>(x);
  }
  //template <typename A> regular_t<A> make_regular(A &&x) REQUIRES(is_ndarray_v<A>) { return std::forward<A>(x); }

} // namespace nda
