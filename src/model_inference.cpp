#include "demucs.hpp"
#include "dsp.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>
#include <Eigen/Dense>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include "demucs.hpp"

Ort::Session demucsonnx::load_model(const std::string htdemucs_model_path) {
    // Initialize ONNX Runtime environment
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "demucs_onnx");

    // Set session options (use defaults)
    Ort::SessionOptions session_options;

    // Create the ONNX Runtime session from the model file
    Ort::Session session(env, htdemucs_model_path.c_str(), session_options);

    return session;
}

// Function to do reflection padding
static void reflect_padding(Eigen::MatrixXf &padded_mix,
                            const Eigen::MatrixXf &mix, int left_padding,
                            int right_padding)
{
    // Assumes padded_mix has size (2, N + left_padding + right_padding)
    // Assumes mix has size (2, N)

    int N = mix.cols(); // The original number of columns

    // Copy the original mix into the middle of padded_mix
    padded_mix.block(0, left_padding, 2, N) = mix;

    // Reflect padding on the left
    for (int i = 0; i < left_padding; ++i)
    {
        padded_mix.block(0, left_padding - 1 - i, 2, 1) = mix.block(0, i, 2, 1);
    }

    // Reflect padding on the right
    for (int i = 0; i < right_padding; ++i)
    {
        padded_mix.block(0, N + left_padding + i, 2, 1) =
            mix.block(0, N - 1 - i, 2, 1);
    }
}

static Ort::Value ConvertColMajorToONNX(
    const Eigen::Tensor3dXf &col_tensor,
    Ort::AllocatorWithDefaultOptions &allocator,
    const std::vector<int64_t> &shape) {
    // Calculate the total size
    size_t size = col_tensor.size();

    // Ensure that the size matches the product of the shape dimensions
    size_t expected_size = 1;
    for (auto dim : shape) {
        expected_size *= dim;
    }
    if (size != expected_size) {
        throw std::runtime_error("Size mismatch between tensor data and shape.");
    }

    // Swap layout to RowMajor and shuffle dimensions
    Eigen::array<int, 3> shuffle_dims = {2, 1, 0};
    Eigen::Tensor<float, 3, Eigen::RowMajor> row_tensor =
        col_tensor.swap_layout().shuffle(shuffle_dims).eval();

    // Create tensor with allocated memory
    Ort::Value ret = Ort::Value::CreateTensor<float>(
        allocator,
        shape.data(),
        shape.size());

    // Copy data into tensor
    float* tensor_data = ret.GetTensorMutableData<float>();
    std::memcpy(tensor_data, row_tensor.data(), size * sizeof(float));

    return ret;
}

template <int Rank>
static Eigen::Tensor<float, Rank, Eigen::ColMajor> ConvertONNXToColMajor(
    Ort::Value &value,
    const std::vector<int64_t>& shape)
{
    // Ensure shape has the expected rank
    if (shape.size() != Rank) {
        throw std::runtime_error("Shape rank does not match expected rank.");
    }

    // Map the ONNX tensor to an Eigen tensor with the appropriate dimensions
    Eigen::array<Eigen::Index, Rank> dims;
    for (int i = 0; i < Rank; ++i) {
        dims[i] = static_cast<Eigen::Index>(shape[i]);
    }

    // Map to Eigen row-major tensor
    const float* output_data = value.GetTensorMutableData<float>();
    Eigen::TensorMap<Eigen::Tensor<const float, Rank, Eigen::RowMajor>> row_tensor_map(output_data, dims);

    // shuffle dims to reverse order after swap_layout
    Eigen::array<int, Rank> shuffle_dims;
    for (int i = 0; i < Rank; ++i) {
        shuffle_dims[i] = Rank - 1 - i;
    }

    // Swap layout to ColMajor
    //return row_tensor_map.swap_layout().shuffle(shuffle_dims);
    // Swap layout to ColMajor and shuffle dimensions
    Eigen::Tensor<float, Rank, Eigen::ColMajor> result_tensor =
        row_tensor_map.swap_layout().shuffle(shuffle_dims).eval();

    return result_tensor;
}

void RunONNXInferenceWithColToRow(Ort::Session &model,
                                  const Eigen::Tensor3dXf &x,
                                  const Eigen::Tensor3dXf &xt,
                                  Eigen::Tensor5dXf &x_out_onnx,
                                  Eigen::Tensor4dXf &xt_out_onnx) {
    Ort::AllocatorWithDefaultOptions allocator;

    // Retrieve expected input shapes from the model
    auto input0_shape = model.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape(); // For xt
    auto input1_shape = model.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape(); // For x

    // Prepare ONNX input tensors by converting ColMajor to RowMajor with correct shapes
    Ort::Value xt_tensor = ConvertColMajorToONNX(xt, allocator, input0_shape); // For input 0
    Ort::Value x_tensor = ConvertColMajorToONNX(x, allocator, input1_shape);   // For input 1

    // Store allocated strings in vectors to prevent dangling pointers
    std::vector<Ort::AllocatedStringPtr> input_name_allocs;
    input_name_allocs.push_back(model.GetInputNameAllocated(0, allocator));
    input_name_allocs.push_back(model.GetInputNameAllocated(1, allocator));

    std::vector<const char*> input_names;
    input_names.push_back(input_name_allocs[0].get());
    input_names.push_back(input_name_allocs[1].get());

    std::vector<Ort::AllocatedStringPtr> output_name_allocs;
    output_name_allocs.push_back(model.GetOutputNameAllocated(0, allocator));
    output_name_allocs.push_back(model.GetOutputNameAllocated(1, allocator));

    std::vector<const char*> output_names;
    output_names.push_back(output_name_allocs[0].get());
    output_names.push_back(output_name_allocs[1].get());

    // Run inference
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(xt_tensor)); // xt corresponds to input 0
    input_tensors.push_back(std::move(x_tensor));  // x corresponds to input 1

    // Ensure RunOptions is properly defined
    Ort::RunOptions run_options;

    // Run the model
    auto output_tensors = model.Run(run_options,
                                    input_names.data(),
                                    input_tensors.data(),
                                    input_tensors.size(),
                                    output_names.data(),
                                    output_names.size());

    // Retrieve expected output shapes
    auto output0_shape = model.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto output1_shape = model.GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();

    // Retrieve actual output shapes from the tensors
    std::vector<int64_t> actual_output0_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> actual_output1_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

    // Convert RowMajor ONNX outputs back to ColMajor Eigen tensors
    x_out_onnx = ConvertONNXToColMajor<5>(output_tensors[0], actual_output0_shape);
    xt_out_onnx = ConvertONNXToColMajor<4>(output_tensors[1], actual_output1_shape);
}

// run core demucs inference using onnx
void demucsonnx::model_inference(
    Ort::Session &model,
    struct demucsonnx::demucs_segment_buffers &buffers,
    struct demucsonnx::stft_buffers &stft_buf,
    ProgressCallback cb, float current_progress,
    float segment_progress)
{
    // pad buffers.pad on the left, reflect
    // pad buffers.pad_end on the right, reflect
    // copy buffers.mix into buffers.padded_mix with reflect padding as above
    reflect_padding(buffers.padded_mix, buffers.mix, buffers.pad,
                    buffers.pad_end);

    // copy buffers.padded_mix into stft_buf.waveform
    stft_buf.waveform = buffers.padded_mix;

    // let's get a stereo complex spectrogram first
    demucsonnx::stft(stft_buf);

    // remove 2: 2 + le of stft
    // same behavior as _spec in the python apply.py code
    buffers.z = stft_buf.spec.slice(
        Eigen::array<int, 3>{0, 0, 2},
        Eigen::array<int, 3>{2, (int)stft_buf.spec.dimension(1),
                             (int)stft_buf.spec.dimension(2) - 4});

    // x = mag = z.abs(), but for CaC we're simply stacking the complex
    // spectrogram along the channel dimension
    for (int i = 0; i < buffers.z.dimension(0); ++i)
    {
        // limiting to j-1 because we're dropping 2049 to 2048 bins
        for (int j = 0; j < buffers.z.dimension(1) - 1; ++j)
        {
            for (int k = 0; k < buffers.z.dimension(2); ++k)
            {
                buffers.x(2 * i, j, k) = buffers.z(i, j, k).real();
                buffers.x(2 * i + 1, j, k) = buffers.z(i, j, k).imag();
            }
        }
    }

    // copy buffers.padded_mix into buffers.xt
    // adding a leading dimension of (1, )

    // prepare time branch input by copying buffers.mix into buffers.xt(0, ...)
    for (int i = 0; i < buffers.mix.rows(); ++i)
    {
        for (int j = 0; j < buffers.mix.cols(); ++j)
        {
            buffers.xt(0, i, j) = buffers.mix(i, j);
        }
    }

    // now we have the stft, apply the core demucs inference
    // (where we removed the stft/istft to successfully convert to ONNX)
    // Apply ONNX inference with col-major to row-major translation
    RunONNXInferenceWithColToRow(model, buffers.x, buffers.xt, buffers.x_out_onnx, buffers.xt_out_onnx);

    std::cout << "ONNX inference completed." << std::endl;

    const int nb_out_sources = 4;

    // 4 sources, 2 channels, N samples
    std::vector<Eigen::MatrixXf> xt_3d = {
        Eigen::MatrixXf(2, buffers.xt.dimension(2)),
        Eigen::MatrixXf(2, buffers.xt.dimension(2)),
        Eigen::MatrixXf(2, buffers.xt.dimension(2)),
        Eigen::MatrixXf(2, buffers.xt.dimension(2))};

    // distribute the channels of buffers.x into x_4d
    // in pytorch it's (16, 2048, 336) i.e. (chan, freq, time)
    // then apply `.view(4, -1, freq, time)

    // x_out_onnx is already x4d

    // let's also copy buffers.xt into xt_4d
    for (int s = 0; s < nb_out_sources; ++s)
    { // loop over 4 sources
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < buffers.xt_out_onnx.dimension(3); ++j)
            {
                xt_3d[s](i, j) = buffers.xt_out_onnx(0, s, i, j);
            }
        }
    }

    // If `cac` is True, `m` is actually a full spectrogram and `z` is ignored.
    // undo complex-as-channels by splitting the 2nd dim of x_4d into (2, 2)
    for (int source = 0; source < nb_out_sources; ++source)
    {
        Eigen::Tensor3dXcf z_target = Eigen::Tensor3dXcf(
            2, buffers.x.dimension(1), buffers.x.dimension(2));

        // in the CaC case, we're simply unstacking the complex
        // spectrogram from the channel dimension
        for (int i = 0; i < buffers.z.dimension(0); ++i)
        {
            // limiting to j-1 because we're dropping 2049 to 2048 bins
            for (int j = 0; j < buffers.z.dimension(1) - 1; ++j)
            {
                for (int k = 0; k < buffers.z.dimension(2); ++k)
                {
                    // buffers.x(2*i, j, k) = buffers.z(i, j, k).real();
                    // buffers.x(2*i + 1, j, k) = buffers.z(i, j, k).imag();
                    z_target(i, j, k) =
                        std::complex<float>(buffers.x_out_onnx(0, source, 2 * i, j, k),
                                            buffers.x_out_onnx(0, source, 2 * i + 1, j, k));
                }
            }
        }

        // need to re-pad 2: 2 + le on spectrogram
        // opposite of this
        // buffers.z = stft_buf.spec.slice(Eigen::array<int, 3>{0, 0, 2},
        //         Eigen::array<int, 3>{2, (int)stft_buf.spec.dimension(1),
        //         (int)stft_buf.spec.dimension(2) - 4});
        // Add padding to spectrogram

        Eigen::array<std::pair<int, int>, 3> paddings = {
            std::make_pair(0, 0), std::make_pair(0, 1), std::make_pair(2, 2)};
        Eigen::Tensor3dXcf z_target_padded =
            z_target.pad(paddings, std::complex<float>(0.0f, 0.0f));

        stft_buf.spec = z_target_padded;

        demucsonnx::istft(stft_buf);

        // now we have waveform from istft(x), the frequency branch
        // that we need to sum with xt, the time branch
        Eigen::MatrixXf padded_waveform = stft_buf.waveform;

        // undo the reflect pad 1d by copying padded_mix into mix
        // from range buffers.pad:buffers.pad + buffers.segment_samples
        Eigen::MatrixXf unpadded_waveform =
            padded_waveform.block(0, buffers.pad, 2, buffers.segment_samples);

        // sum with xt
        unpadded_waveform += xt_3d[source];

        // copy target waveform into all 4 dims of targets_out
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < buffers.mix.cols(); ++k)
            {
                buffers.targets_out(source, j, k) = unpadded_waveform(j, k);
            }
        }
    }
}
