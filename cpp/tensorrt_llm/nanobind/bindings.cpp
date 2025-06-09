/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/nanobind/executor/bindings.h"
#include <nanobind/nanobind.h>
#include <torch/extension.h>
#include <vector>

namespace nano = nanobind;

NB_MODULE(TRTLLM_NANOBIND_MODULE, m)
{
    m.doc() = "TensorRT-LLM Python bindings for C++ runtime using nanobind";

    // Create submodule for executor bindings.
    auto mExecutor = m.def_submodule("executor", "Executor bindings");

    tensorrt_llm::nanobind::executor::initBindings(mExecutor);
}
