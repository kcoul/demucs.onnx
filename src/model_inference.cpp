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
// this is the model model baked into a header file
#include "htdemucs.ort.h"

Ort::Session demucsonnx::load_model() {
    // Initialize ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "demucs_onnx");

    // Set session options (use defaults)
    Ort::SessionOptions session_options;

    // Create the ONNX Runtime session from the in-memory ORT model
    Ort::Session session(env, htdemucs_ort_start, htdemucs_ort_size, session_options);

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

// Helper function to convert a ColMajor Eigen Tensor to RowMajor ONNX tensor
static Ort::Value ConvertColMajorToONNX(const Eigen::Tensor3dXf &col_tensor, Ort::MemoryInfo &memory_info) {
    // Flatten Eigen tensor in column-major order and convert to row-major layout in 1D vector
    std::vector<float> row_major_data(col_tensor.size());
    Eigen::Tensor3dRowMajorXf row_tensor = col_tensor.swap_layout();  // Swap to RowMajor layout
    std::memcpy(row_major_data.data(), row_tensor.data(), row_tensor.size() * sizeof(float));

    // Define shape based on the dimensions of the input tensor
    std::vector<int64_t> shape = {1, col_tensor.dimension(0), col_tensor.dimension(1), col_tensor.dimension(2)};

    // Create and return the ONNX Runtime tensor
    return Ort::Value::CreateTensor<float>(memory_info, row_major_data.data(), row_major_data.size(), shape.data(), shape.size());
}

// Helper function to convert ONNX RowMajor tensor output back to Eigen ColMajor tensor
static Eigen::Tensor3dXf ConvertONNXToColMajor(Ort::Value &value, const Eigen::Tensor3dXf::Dimensions &dims) {
    // Retrieve data from ONNX tensor, which is row-major
    const float* output_data = value.GetTensorMutableData<float>();

    // Map to Eigen row-major tensor
    Eigen::TensorMap<Eigen::Tensor<const float, 3, Eigen::RowMajor>> row_tensor_map(output_data, dims[0], dims[1], dims[2]);

    // Convert to column-major tensor
    Eigen::Tensor3dXf col_tensor(dims);
    col_tensor = row_tensor_map.swap_layout();  // Swap layout to ColMajor
    return col_tensor;
}

static void RunONNXInferenceWithColToRow(Ort::Session &model, const Eigen::Tensor3dXf &x, const Eigen::Tensor3dXf &xt, Eigen::Tensor3dXf &x_out, Eigen::Tensor3dXf &xt_out) {
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Prepare ONNX input tensors by converting ColMajor to RowMajor
    Ort::Value x_tensor = ConvertColMajorToONNX(x, memory_info);
    Ort::Value xt_tensor = ConvertColMajorToONNX(xt, memory_info);

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
    input_tensors.push_back(std::move(x_tensor));
    input_tensors.push_back(std::move(xt_tensor));

    // Ensure RunOptions is properly defined
    Ort::RunOptions run_options;

    // Run the model
    auto output_tensors = model.Run(run_options,
                                    input_names.data(),
                                    input_tensors.data(),
                                    input_tensors.size(),
                                    output_names.data(),
                                    output_names.size());

    // Convert RowMajor ONNX outputs back to ColMajor Eigen tensors
    x_out = ConvertONNXToColMajor(output_tensors[0], x_out.dimensions());
    xt_out = ConvertONNXToColMajor(output_tensors[1], xt_out.dimensions());
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

    std::ostringstream ss;
    // print z shape
    ss << "buffers.z: " << buffers.z.dimension(0) << ", "
       << buffers.z.dimension(1) << ", " << buffers.z.dimension(2);
    cb(current_progress + 0.0f, ss.str());
    ss.str("");

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

    // x shape is complex*chan, nb_frames, nb_bins (2048)
    // using CaC (complex-as-channels)
    // print x shape
    ss << "buffers.x: " << buffers.x.dimension(0) << ", "
       << buffers.x.dimension(1) << ", " << buffers.x.dimension(2);
    cb(current_progress + 0.0f, ss.str());
    ss.str("");

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

    // print it like the above
    ss << "buffers.xt: " << buffers.xt.dimension(0) << ", "
       << buffers.xt.dimension(1) << ", " << buffers.xt.dimension(2);
    cb(current_progress + 0.0f, ss.str());
    ss.str("");

    // now we have the stft, apply the core demucs inference
    // (where we removed the stft/istft to successfully convert to ONNX)

    Ort::AllocatorWithDefaultOptions allocator;

    // Get and print input tensor information
    size_t num_inputs = model.GetInputCount();
    std::cout << "Number of inputs: " << num_inputs << std::endl;

    for (size_t i = 0; i < num_inputs; ++i) {
        std::string input_name = model.GetInputNameAllocated(i, allocator).get();
        std::cout << "Input " << i << " name: " << input_name << std::endl;

        Ort::TypeInfo type_info = model.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        // Print element type
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::cout << "Input " << i << " type: " << type << std::endl;

        // Print shape
        std::vector<int64_t> input_shape = tensor_info.GetShape();
        std::cout << "Input " << i << " shape: ";
        for (auto dim : input_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }

    // Get and print output tensor information
    size_t num_outputs = model.GetOutputCount();
    std::cout << "Number of outputs: " << num_outputs << std::endl;

    for (size_t i = 0; i < num_outputs; ++i) {
        std::string output_name = model.GetOutputNameAllocated(i, allocator).get();
        std::cout << "Output " << i << " name: " << output_name << std::endl;

        Ort::TypeInfo type_info = model.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        // Print element type
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::cout << "Output " << i << " type: " << type << std::endl;

        // Print shape
        std::vector<int64_t> output_shape = tensor_info.GetShape();
        std::cout << "Output " << i << " shape: ";
        for (auto dim : output_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }

    // Apply ONNX inference with col-major to row-major translation
    RunONNXInferenceWithColToRow(model, buffers.x, buffers.xt, buffers.x_out, buffers.xt_out);

    // create output tensor

    const int nb_out_sources = 4;

    // 4 sources, 2 channels * 2 complex channels (real+imag), F bins, T frames
    Eigen::Tensor4dXf x_4d = Eigen::Tensor4dXf(
        nb_out_sources, 4, buffers.x.dimension(1), buffers.x.dimension(2));

    // 4 sources, 2 channels, N samples
    std::vector<Eigen::MatrixXf> xt_3d = {
        Eigen::MatrixXf(2, buffers.xt.dimension(2)),
        Eigen::MatrixXf(2, buffers.xt.dimension(2)),
        Eigen::MatrixXf(2, buffers.xt.dimension(2)),
        Eigen::MatrixXf(2, buffers.xt.dimension(2))};

    // distribute the channels of buffers.x into x_4d
    // in pytorch it's (16, 2048, 336) i.e. (chan, freq, time)
    // then apply `.view(4, -1, freq, time)

    // implement above logic in Eigen C++
    // copy buffers.x into x_4d
    // apply opposite of
    // buffers.x(i, j, k) = (buffers.x(i, j, k) - mean) / (epsilon + std_);
    for (int s = 0; s < nb_out_sources; ++s)
    { // loop over 4 sources
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < buffers.x.dimension(1); ++j)
            {
                for (int k = 0; k < buffers.x.dimension(2); ++k)
                {
                    x_4d(s, i, j, k) = buffers.x_out(s * 4 + i, j, k);
                }
            }
        }
    }

    // let's also copy buffers.xt into xt_4d
    for (int s = 0; s < nb_out_sources; ++s)
    { // loop over 4 sources
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < buffers.xt.dimension(2); ++j)
            {
                xt_3d[s](i, j) = buffers.xt_out(0, s * 2 + i, j);
            }
        }
    }

    // If `cac` is True, `m` is actually a full spectrogram and `z` is ignored.
    // undo complex-as-channels by splitting the 2nd dim of x_4d into (2, 2)
    for (int source = 0; source < nb_out_sources; ++source)
    {
        Eigen::Tensor3dXcf x_target = Eigen::Tensor3dXcf(
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
                    x_target(i, j, k) =
                        std::complex<float>(x_4d(source, 2 * i, j, k),
                                            x_4d(source, 2 * i + 1, j, k));
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
        Eigen::Tensor3dXcf x_target_padded =
            x_target.pad(paddings, std::complex<float>(0.0f, 0.0f));

        stft_buf.spec = x_target_padded;

        demucsonnx::istft(stft_buf);

        // now we have waveform from istft(x), the frequency branch
        // that we need to sum with xt, the time branch
        Eigen::MatrixXf padded_waveform = stft_buf.waveform;

        // undo the reflect pad 1d by copying padded_mix into mix
        // from range buffers.pad:buffers.pad + buffers.segment_samples
        Eigen::MatrixXf unpadded_waveform =
            padded_waveform.block(0, buffers.pad, 2, buffers.segment_samples);

        // sum with xt
        // choose a different source to sum with in case
        // they're in different orders...
        unpadded_waveform += xt_3d[source];

        ss << "mix: " << buffers.mix.rows() << ", " << buffers.mix.cols();
        cb(current_progress + segment_progress, ss.str());
        ss.str("");

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
