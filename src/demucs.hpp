#ifndef MODEL_HPP
#define MODEL_HPP

#include "dsp.hpp"
#include "tensor.hpp"
#include <Eigen/Dense>
#include <array>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

namespace demucsonnx
{

// Define a type for your callback function
using ProgressCallback = std::function<void(float, const std::string &)>;

const int FREQ_BRANCH_LEN = 336;
const int TIME_BRANCH_LEN_IN = 343980;

struct demucs_segment_buffers
{
    int segment_samples;
    int le;
    int pad;
    int pad_end;
    int padded_segment_samples;
    int nb_stft_frames;
    int nb_stft_bins;

    Eigen::MatrixXf mix;
    Eigen::Tensor3dXf targets_out;
    Eigen::MatrixXf padded_mix;
    Eigen::Tensor3dXcf z;

    Eigen::Tensor3dXf x;     // input
    Eigen::Tensor3dXf xt;     // input

    Eigen::Tensor3dXf x_out; // output
    Eigen::Tensor3dXf xt_out; // output

    // constructor for demucs_segment_buffers that takes int parameters

    // let's do pesky precomputing of the signal repadding to 1/4 hop
    // for time and frequency alignment
    demucs_segment_buffers(int nb_channels, int segment_samples, int nb_sources)
        : segment_samples(segment_samples),
          le(int(std::ceil((float)segment_samples / (float)FFT_HOP_SIZE))),
          pad(std::floor((float)FFT_HOP_SIZE / 2.0f) * 3),
          pad_end(pad + le * FFT_HOP_SIZE - segment_samples),
          padded_segment_samples(segment_samples + pad + pad_end),
          nb_stft_frames(segment_samples / demucsonnx::FFT_HOP_SIZE + 1),
          nb_stft_bins(demucsonnx::FFT_WINDOW_SIZE / 2 + 1),
          mix(nb_channels, segment_samples),
          targets_out(nb_sources, nb_channels, segment_samples),
          padded_mix(nb_channels, padded_segment_samples),
          z(nb_channels, nb_stft_bins, nb_stft_frames),
          // complex-as-channels implies 2*nb_channels for real+imag
          x(2 * nb_channels, nb_stft_bins - 1, nb_stft_frames),
          xt(1, nb_channels, segment_samples),
          x_out(nb_sources * 2 * nb_channels, nb_stft_bins - 1, nb_stft_frames),
          xt_out(1, nb_sources * nb_channels, segment_samples){};
};

Ort::Session load_model(const std::string htdemucs_model_path);

const float SEGMENT_LEN_SECS = 7.8;      // 8 seconds, the demucs chunk size
const float SEGMENT_OVERLAP_SECS = 0.25; // 0.25 overlap
const float MAX_SHIFT_SECS = 0.5;        // max shift
const float OVERLAP = 0.25;              // overlap between segments
const float TRANSITION_POWER = 1.0;      // transition between segments

Eigen::Tensor3dXf demucs_inference(Ort::Session &model,
                                   const Eigen::MatrixXf &full_audio,
                                   ProgressCallback cb);

void model_inference(Ort::Session &model,
                     struct demucsonnx::demucs_segment_buffers &buffers,
                     struct demucsonnx::stft_buffers &stft_buf,
                     ProgressCallback cb, float current_progress,
                     float segment_progress);
} // namespace demucsonnx

#endif // MODEL_HPP
