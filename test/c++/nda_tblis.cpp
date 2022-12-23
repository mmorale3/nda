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

#include <type_traits>
#include "test_common.hpp"

#include <nda/tensor.hpp>
//#include <nda/clef/literals.hpp>

using nda::F_layout;
//using namespace clef::literals;

//----------------------------

template <typename value_t, typename Layout>
void test_contract() {

  using other_layout = std::conditional_t<std::is_same_v<Layout,C_layout>,F_layout,C_layout>;
  { // ik,kj->ij
    matrix<value_t, Layout> M1{{0, 1}, {1, 2}}, M2{{1, 1}, {1, 1}}, M3{{1, 0}, {0, 1}};
    nda::tensor::contract(1.0, M1, "ik", M2, "kj", 1.0, M3, "ij");

    EXPECT_ARRAY_NEAR(M1, nda::matrix<value_t>{{0, 1}, {1, 2}});
    EXPECT_ARRAY_NEAR(M2, nda::matrix<value_t>{{1, 1}, {1, 1}});
    EXPECT_ARRAY_NEAR(M3, nda::matrix<value_t>{{2, 1}, {3, 4}});
  }

  { 
    nda::array<value_t, 3, Layout> M1{{{0,1},{2,3}},{{4,5},{6,7}}};
    nda::array<value_t, 3, Layout> M2{{{0,2},{4,6}},{{8,10},{12,14}}};
    nda::array<value_t, 3, Layout> M3(2,2,2);
    nda::array<value_t, 2, Layout> M4(2,2);
    nda::array<value_t, 1, Layout> M5(2);
    M3()=0;
    M4()=0;
    M5()=0;

    // ijk,ikl->ijl 
    nda::tensor::contract(1.0, M1, "ijk", M2, "ikl", 0.0, M3, "ijl");
    EXPECT_ARRAY_NEAR(M3, nda::array<value_t, 3>{{{4,6},{12,22}},{{92,110},{132,158}}});

    // ikj,kli->lij 
    nda::tensor::contract(1.0, M1, "ikj", M2, "kli", 0.0, M3, "lij");
    EXPECT_ARRAY_NEAR(M3, nda::array<value_t, 3>{{{16,24},{68,80}},{{24,40},{108,128}}});

    // ijk,klj->li 
    nda::tensor::contract(1.0, M1, "ijk", M2, "klj", 0.0, M4, "li");
    EXPECT_ARRAY_NEAR(M4, nda::array<value_t, 2>{{42,122},{66,210}});

    // ijk,kij->i 
    nda::tensor::contract(1.0, M1, "ijk", M2, "kij", 0.0, M5, "i");
    EXPECT_ARRAY_NEAR(M5, nda::array<value_t, 1>{42,210});
  }

  // mixed layouts
  { // ik,kj->ij
    matrix<value_t, Layout> M1{{0, 1}, {1, 2}}, M3{{1, 0}, {0, 1}};
    matrix<value_t, other_layout> M2{{1, 1}, {1, 1}};
    nda::tensor::contract(1.0, M1, "ik", M2, "kj", 1.0, M3, "ij");
    
    EXPECT_ARRAY_NEAR(M1, nda::matrix<value_t>{{0, 1}, {1, 2}});
    EXPECT_ARRAY_NEAR(M2, nda::matrix<value_t>{{1, 1}, {1, 1}});
    EXPECT_ARRAY_NEAR(M3, nda::matrix<value_t>{{2, 1}, {3, 4}});
  }

  {
    nda::array<value_t, 3, Layout> M1{{{0,1},{2,3}},{{4,5},{6,7}}};
    nda::array<value_t, 3, other_layout> M2{{{0,2},{4,6}},{{8,10},{12,14}}};
    nda::array<value_t, 3, Layout> M3(2,2,2);
    nda::array<value_t, 2, Layout> M4(2,2);
    nda::array<value_t, 1, Layout> M5(2);
    M3()=0;
    M4()=0;
    M5()=0;

    // ijk,ikl->ijl 
    nda::tensor::contract(1.0, M1, "ijk", M2, "ikl", 0.0, M3, "ijl");
    EXPECT_ARRAY_NEAR(M3, nda::array<value_t, 3>{{{4,6},{12,22}},{{92,110},{132,158}}});

    // ikj,kli->lij 
    nda::tensor::contract(1.0, M1, "ikj", M2, "kli", 0.0, M3, "lij");
    EXPECT_ARRAY_NEAR(M3, nda::array<value_t, 3>{{{16,24},{68,80}},{{24,40},{108,128}}});

    // ijk,klj->li 
    nda::tensor::contract(1.0, M1, "ijk", M2, "klj", 0.0, M4, "li");
    EXPECT_ARRAY_NEAR(M4, nda::array<value_t, 2>{{42,122},{66,210}});

    // ijk,kij->i 
    nda::tensor::contract(1.0, M1, "ijk", M2, "kij", 0.0, M5, "i");
    EXPECT_ARRAY_NEAR(M5, nda::array<value_t, 1>{42,210});
  }
     
}

TEST(TENSOR, contract) { test_contract<double, C_layout>(); }     //NOLINT
TEST(TENSOR, contractF) { test_contract<double, F_layout>(); }    //NOLINT
TEST(TENSOR, zcontract) { test_contract<dcomplex, C_layout>(); }  //NOLINT
TEST(TENSOR, zcontractF) { test_contract<dcomplex, F_layout>(); } //NOLINT
