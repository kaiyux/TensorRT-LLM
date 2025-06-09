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

#include "bindings.h"
#include "request.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"

#include <nanobind/nanobind.h>

#include <optional>

namespace nano = nanobind;
namespace tle = tensorrt_llm::executor;
using SizeType32 = tle::SizeType32;

namespace tensorrt_llm::nanobind::executor
{

void initBindings(nanobind::module_& m)
{
    m.attr("__version__") = tle::version();
    tensorrt_llm::nanobind::executor::initRequestBindings(m);
}

} // namespace tensorrt_llm::nanobind::executor
