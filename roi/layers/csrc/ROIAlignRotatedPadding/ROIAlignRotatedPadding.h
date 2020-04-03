// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <torch/types.h>

namespace detectron2 {

at::Tensor ROIAlignRotatedPadding_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio);

at::Tensor ROIAlignRotatedPadding_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio);

#ifdef WITH_CUDA
at::Tensor ROIAlignRotatedPadding_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio);

at::Tensor ROIAlignRotatedPadding_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio);
#endif

// Interface for Python
inline at::Tensor ROIAlignRotatedPadding_forward(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlignRotatedPadding_forward_cuda(
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ROIAlignRotatedPadding_forward_cpu(
      input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
}

inline at::Tensor ROIAlignRotatedPadding_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlignRotatedPadding_backward_cuda(
        grad,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        batch_size,
        channels,
        height,
        width,
        sampling_ratio);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ROIAlignRotatedPadding_backward_cpu(
      grad,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      batch_size,
      channels,
      height,
      width,
      sampling_ratio);
}

} // namespace detectron2
