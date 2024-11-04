#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <limits>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

namespace Eigen
{
// define Tensor3dXf, Tensor3dXcf for spectrograms etc.
typedef Tensor<float, 5> Tensor5dXf;
typedef Tensor<float, 4> Tensor4dXf;
typedef Tensor<float, 3> Tensor3dXf;
typedef Tensor<float, 3, RowMajor> Tensor3dRowMajorXf;
typedef Tensor<float, 2> Tensor2dXf;
typedef Tensor<float, 1> Tensor1dXf;
typedef Tensor<std::complex<float>, 3> Tensor3dXcf;
} // namespace Eigen

#endif // TENSOR_HPP
