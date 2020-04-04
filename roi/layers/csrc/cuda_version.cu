// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

#include <cuda_runtime_api.h>

namespace roi {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace roi
