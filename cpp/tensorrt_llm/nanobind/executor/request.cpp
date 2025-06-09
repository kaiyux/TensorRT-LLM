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

#include "request.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include <nanobind/nanobind.h>

#include <optional>
#include <vector>

namespace nano = nanobind;
namespace tle = tensorrt_llm::executor;
using Tensor = tle::Tensor;
using SizeType32 = tle::SizeType32;
using FloatType = tle::FloatType;
using VecTokens = tle::VecTokens;
using IdType = tle::IdType;
using VecTokenExtraIds = tle::VecTokenExtraIds;

namespace tensorrt_llm::nanobind::executor
{

void initRequestBindings(nano::module_& m)
{
    nano::enum_<tle::RequestType>(m, "RequestType")
        .value("REQUEST_TYPE_CONTEXT_AND_GENERATION", tle::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION)
        .value("REQUEST_TYPE_CONTEXT_ONLY", tle::RequestType::REQUEST_TYPE_CONTEXT_ONLY)
        .value("REQUEST_TYPE_GENERATION_ONLY", tle::RequestType::REQUEST_TYPE_GENERATION_ONLY);

    nano::enum_<tle::FinishReason>(m, "FinishReason")
        .value("NOT_FINISHED", tle::FinishReason::kNOT_FINISHED)
        .value("END_ID", tle::FinishReason::kEND_ID)
        .value("STOP_WORDS", tle::FinishReason::kSTOP_WORDS)
        .value("LENGTH", tle::FinishReason::kLENGTH)
        .value("TIMED_OUT", tle::FinishReason::kTIMED_OUT)
        .value("CANCELLED", tle::FinishReason::kCANCELLED);

    nano::enum_<tle::KvCacheTransferMode>(m, "KvCacheTransferMode")
        .value("DRAM", tle::KvCacheTransferMode::DRAM)
        .value("GDS", tle::KvCacheTransferMode::GDS)
        .value("POSIX_DEBUG_FALLBACK", tle::KvCacheTransferMode::POSIX_DEBUG_FALLBACK);

    auto samplingConfigGetstate = [](tle::SamplingConfig const& self)
    {
        return nano::make_tuple(self.getBeamWidth(), self.getTopK(), self.getTopP(), self.getTopPMin(),
            self.getTopPResetIds(), self.getTopPDecay(), self.getSeed(), self.getTemperature(), self.getMinTokens(),
            self.getBeamSearchDiversityRate(), self.getRepetitionPenalty(), self.getPresencePenalty(),
            self.getFrequencyPenalty(), self.getLengthPenalty(), self.getEarlyStopping(), self.getNoRepeatNgramSize(),
            self.getNumReturnSequences(), self.getMinP(), self.getBeamWidthArray());
    };
    auto samplingConfigSetstate = [](nano::tuple const& state)
    {
        if (state.size() != 19)
        {
            throw std::runtime_error("Invalid SamplingConfig state!");
        }
        return tle::SamplingConfig(nano::cast<SizeType32>(state[0]),      // BeamWidth
            nano::cast<std::optional<SizeType32>>(state[1]),              // TopK
            nano::cast<std::optional<FloatType>>(state[2]),               // TopP
            nano::cast<std::optional<FloatType>>(state[3]),               // TopPMin
            nano::cast<std::optional<tle::TokenIdType>>(state[4]),        // TopPResetIds
            nano::cast<std::optional<FloatType>>(state[5]),               // TopPDecay
            nano::cast<std::optional<tle::RandomSeedType>>(state[6]),     // Seed
            nano::cast<std::optional<FloatType>>(state[7]),               // Temperature
            nano::cast<std::optional<SizeType32>>(state[8]),              // MinTokens
            nano::cast<std::optional<FloatType>>(state[9]),               // BeamSearchDiversityRate
            nano::cast<std::optional<FloatType>>(state[10]),              // RepetitionPenalty
            nano::cast<std::optional<FloatType>>(state[11]),              // PresencePenalty
            nano::cast<std::optional<FloatType>>(state[12]),              // FrequencyPenalty
            nano::cast<std::optional<FloatType>>(state[13]),              // LengthPenalty
            nano::cast<std::optional<SizeType32>>(state[14]),             // EarlyStopping
            nano::cast<std::optional<SizeType32>>(state[15]),             // NoRepeatNgramSize
            nano::cast<std::optional<SizeType32>>(state[16]),             // NumReturnSequences
            nano::cast<std::optional<FloatType>>(state[17]),              // MinP
            nano::cast<std::optional<std::vector<SizeType32>>>(state[18]) // BeamWidthArray
        );
    };
    nano::class_<tle::SamplingConfig>(m, "SamplingConfig")
        .def(nano::init<tle::SizeType32,
                 std::optional<tle::SizeType32> const&,             // beamWidth
                 std::optional<tle::FloatType> const&,              // topP
                 std::optional<tle::FloatType> const&,              // topPMin
                 std::optional<tle::TokenIdType> const&,            // topPResetIds
                 std::optional<tle::FloatType> const&,              // topPDecay
                 std::optional<tle::RandomSeedType> const&,         // seed
                 std::optional<tle::FloatType> const&,              // temperature
                 std::optional<tle::SizeType32> const&,             // minTokens
                 std::optional<tle::FloatType> const&,              // beamSearchDiversityRate
                 std::optional<tle::FloatType> const&,              // repetitionPenalty
                 std::optional<tle::FloatType> const&,              // presencePenalty
                 std::optional<tle::FloatType> const&,              // frequencyPenalty
                 std::optional<tle::FloatType> const&,              // lengthPenalty
                 std::optional<tle::SizeType32> const&,             // earlyStopping
                 std::optional<tle::SizeType32> const&,             // noRepeatNgramSize
                 std::optional<tle::SizeType32> const&,             // numReturnSequences
                 std::optional<tle::FloatType> const&,              // minP
                 std::optional<std::vector<tle::SizeType32>> const& // beamWidthArray
                 >(),
            // clang-format off
            nano::arg("beam_width") = 1,
            nano::kw_only(),
            nano::arg("top_k") = nano::none(),
            nano::arg("top_p") = nano::none(),
            nano::arg("top_p_min") = nano::none(),
            nano::arg("top_p_reset_ids") = nano::none(),
            nano::arg("top_p_decay") = nano::none(),
            nano::arg("seed") = nano::none(),
            nano::arg("temperature") = nano::none(),
            nano::arg("min_tokens") = nano::none(),
            nano::arg("beam_search_diversity_rate") = nano::none(),
            nano::arg("repetition_penalty") = nano::none(),
            nano::arg("presence_penalty") = nano::none(),
            nano::arg("frequency_penalty") = nano::none(),
            nano::arg("length_penalty") = nano::none(),
            nano::arg("early_stopping") = nano::none(),
            nano::arg("no_repeat_ngram_size") = nano::none(),
            nano::arg("num_return_sequences") = nano::none(),
            nano::arg("min_p") = nano::none(),
            nano::arg("beam_width_array") = nano::none())           // clang-format on
        .def_prop_rw("beam_width", &tle::SamplingConfig::getBeamWidth, &tle::SamplingConfig::setBeamWidth)
        .def_prop_rw("top_k", &tle::SamplingConfig::getTopK, &tle::SamplingConfig::setTopK)
        .def_prop_rw("top_p", &tle::SamplingConfig::getTopP, &tle::SamplingConfig::setTopP)
        .def_prop_rw("top_p_min", &tle::SamplingConfig::getTopPMin, &tle::SamplingConfig::setTopPMin)
        .def_prop_rw("top_p_reset_ids", &tle::SamplingConfig::getTopPResetIds, &tle::SamplingConfig::setTopPResetIds)
        .def_prop_rw("top_p_decay", &tle::SamplingConfig::getTopPDecay, &tle::SamplingConfig::setTopPDecay)
        .def_prop_rw("seed", &tle::SamplingConfig::getSeed, &tle::SamplingConfig::setSeed)
        .def_prop_rw("temperature", &tle::SamplingConfig::getTemperature, &tle::SamplingConfig::setTemperature)
        .def_prop_rw("min_tokens", &tle::SamplingConfig::getMinTokens, &tle::SamplingConfig::setMinTokens)
        .def_prop_rw("beam_search_diversity_rate", &tle::SamplingConfig::getBeamSearchDiversityRate,
            &tle::SamplingConfig::setBeamSearchDiversityRate)
        .def_prop_rw("repetition_penalty", &tle::SamplingConfig::getRepetitionPenalty,
            &tle::SamplingConfig::setRepetitionPenalty)
        .def_prop_rw("presence_penalty", &tle::SamplingConfig::getPresencePenalty,
            [](tle::SamplingConfig& self, std::optional<FloatType> v) { self.setPresencePenalty(v); })
        .def_prop_rw(
            "frequency_penalty", &tle::SamplingConfig::getFrequencyPenalty, &tle::SamplingConfig::setFrequencyPenalty)
        .def_prop_rw("length_penalty", &tle::SamplingConfig::getLengthPenalty, &tle::SamplingConfig::setLengthPenalty)
        .def_prop_rw("early_stopping", &tle::SamplingConfig::getEarlyStopping, &tle::SamplingConfig::setEarlyStopping)
        .def_prop_rw("no_repeat_ngram_size", &tle::SamplingConfig::getNoRepeatNgramSize,
            &tle::SamplingConfig::setNoRepeatNgramSize)
        .def_prop_rw("num_return_sequences", &tle::SamplingConfig::getNumReturnSequences,
            &tle::SamplingConfig::setNumReturnSequences)
        .def_prop_rw("min_p", &tle::SamplingConfig::getMinP, &tle::SamplingConfig::setMinP)
        .def_prop_rw(
            "beam_width_array", &tle::SamplingConfig::getBeamWidthArray, &tle::SamplingConfig::setBeamWidthArray)
        // .def(nano::pickle(samplingConfigGetstate, samplingConfigSetstate));
        .def("__getstate__", samplingConfigGetstate)
        .def("__setstate__", samplingConfigSetstate);

    auto additionalModelOutputGetstate
        = [](tle::AdditionalModelOutput const& self) { return nano::make_tuple(self.name, self.gatherContext); };
    auto additionalModelOutputSetstate = [](nano::tuple const& state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid AdditionalModelOutput state!");
        }
        return tle::AdditionalModelOutput(nano::cast<std::string>(state[0]), nano::cast<bool>(state[1]));
    };
    nano::class_<tle::AdditionalModelOutput>(m, "AdditionalModelOutput")
        .def(nano::init<std::string, bool>(), nano::arg("name"), nano::arg("gather_context") = false)
        .def_rw("name", &tle::AdditionalModelOutput::name)
        .def_rw("gather_context", &tle::AdditionalModelOutput::gatherContext)
        // .def(nano::pickle(additionalModelOutputGetstate, additionalModelOutputSetstate));
        .def("__getstate__", additionalModelOutputGetstate)
        .def("__setstate__", additionalModelOutputSetstate);

    auto outputConfigGetstate = [](tle::OutputConfig const& self)
    {
        return nano::make_tuple(self.returnLogProbs, self.returnContextLogits, self.returnGenerationLogits,
            self.excludeInputFromOutput, self.returnEncoderOutput, self.returnPerfMetrics, self.additionalModelOutputs);
    };
    auto outputConfigSetstate = [](nano::tuple const& state)
    {
        if (state.size() != 7)
        {
            throw std::runtime_error("Invalid OutputConfig state!");
        }
        return tle::OutputConfig(nano::cast<bool>(state[0]), nano::cast<bool>(state[1]), nano::cast<bool>(state[2]),
            nano::cast<bool>(state[3]), nano::cast<bool>(state[4]), nano::cast<bool>(state[5]),
            nano::cast<std::optional<std::vector<tle::AdditionalModelOutput>>>(state[6]));
    };
    nano::class_<tle::OutputConfig>(m, "OutputConfig")
        .def(nano::init<bool, bool, bool, bool, bool, bool, std::optional<std::vector<tle::AdditionalModelOutput>>>(),
            nano::arg("return_log_probs") = false, nano::arg("return_context_logits") = false,
            nano::arg("return_generation_logits") = false, nano::arg("exclude_input_from_output") = false,
            nano::arg("return_encoder_output") = false, nano::arg("return_perf_metrics") = false,
            nano::arg("additional_model_outputs") = nano::none())
        .def_rw("return_log_probs", &tle::OutputConfig::returnLogProbs)
        .def_rw("return_context_logits", &tle::OutputConfig::returnContextLogits)
        .def_rw("return_generation_logits", &tle::OutputConfig::returnGenerationLogits)
        .def_rw("exclude_input_from_output", &tle::OutputConfig::excludeInputFromOutput)
        .def_rw("return_encoder_output", &tle::OutputConfig::returnEncoderOutput)
        .def_rw("return_perf_metrics", &tle::OutputConfig::returnPerfMetrics)
        .def_rw("additional_model_outputs", &tle::OutputConfig::additionalModelOutputs)
        // .def(nano::pickle(outputConfigGetstate, outputConfigSetstate));
        .def("__getstate__", outputConfigGetstate)
        .def("__setstate__", outputConfigSetstate);

    auto externalDraftTokensConfigGetstate = [](tle::ExternalDraftTokensConfig const& self)
    { return nano::make_tuple(self.getTokens(), self.getLogits(), self.getAcceptanceThreshold()); };
    auto externalDraftTokensConfigSetstate = [](nano::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid ExternalDraftTokensConfig state!");
        }
        return tle::ExternalDraftTokensConfig(nano::cast<VecTokens>(state[0]),
            nano::cast<std::optional<Tensor>>(state[1]), nano::cast<std::optional<FloatType>>(state[2]));
    };
    nano::class_<tle::ExternalDraftTokensConfig>(m, "ExternalDraftTokensConfig")
        .def(nano::init<VecTokens, std::optional<Tensor>, std::optional<FloatType> const&, std::optional<bool>>(),
            nano::arg("tokens"), nano::arg("logits") = nano::none(), nano::arg("acceptance_threshold") = nano::none(),
            nano::arg("fast_logits") = nano::none())
        .def_prop_ro("tokens", &tle::ExternalDraftTokensConfig::getTokens)
        .def_prop_ro("logits", &tle::ExternalDraftTokensConfig::getLogits)
        .def_prop_ro("acceptance_threshold", &tle::ExternalDraftTokensConfig::getAcceptanceThreshold)
        // .def(nano::pickle(externalDraftTokensConfigGetstate, externalDraftTokensConfigSetstate))
        .def("__getstate__", externalDraftTokensConfigGetstate)
        .def("__setstate__", externalDraftTokensConfigSetstate)
        .def_prop_ro("fast_logits", &tle::ExternalDraftTokensConfig::getFastLogits);

    auto promptTuningConfigGetstate = [](tle::PromptTuningConfig const& self)
    { return nano::make_tuple(self.getEmbeddingTable(), self.getInputTokenExtraIds()); };
    auto promptTuningConfigSetstate = [](nano::tuple const& state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid PromptTuningConfig state!");
        }
        return tle::PromptTuningConfig(
            nano::cast<Tensor>(state[0]), nano::cast<std::optional<VecTokenExtraIds>>(state[1]));
    };
    nano::class_<tle::PromptTuningConfig>(m, "PromptTuningConfig")
        .def(nano::init<Tensor, std::optional<VecTokenExtraIds>>(), nano::arg("embedding_table"),
            nano::arg("input_token_extra_ids") = nano::none())
        .def_prop_ro("embedding_table", &tle::PromptTuningConfig::getEmbeddingTable)
        .def_prop_ro("input_token_extra_ids", &tle::PromptTuningConfig::getInputTokenExtraIds)
        // .def(nano::pickle(promptTuningConfigGetstate, promptTuningConfigSetstate));
        .def("__getstate__", promptTuningConfigGetstate)
        .def("__setstate__", promptTuningConfigSetstate);

    auto loraConfigGetstate = [](tle::LoraConfig const& self)
    { return nano::make_tuple(self.getTaskId(), self.getWeights(), self.getConfig()); };
    auto loraConfigSetstate = [](nano::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid LoraConfig state!");
        }
        return tle::LoraConfig(nano::cast<IdType>(state[0]), nano::cast<std::optional<Tensor>>(state[1]),
            nano::cast<std::optional<Tensor>>(state[2]));
    };
    nano::class_<tle::LoraConfig>(m, "LoraConfig")
        .def(nano::init<uint64_t, std::optional<Tensor>, std::optional<Tensor>>(), nano::arg("task_id"),
            nano::arg("weights") = nano::none(), nano::arg("config") = nano::none())
        .def_prop_ro("task_id", &tle::LoraConfig::getTaskId)
        .def_prop_ro("weights", &tle::LoraConfig::getWeights)
        .def_prop_ro("config", &tle::LoraConfig::getConfig)
        // .def(nano::pickle(loraConfigGetstate, loraConfigSetstate));
        .def("__getstate__", loraConfigGetstate)
        .def("__setstate__", loraConfigSetstate);

    auto MropeConfigGetstate = [](tle::MropeConfig const& self)
    { return nano::make_tuple(self.getMRopeRotaryCosSin(), self.getMRopePositionDeltas()); };
    auto MropeConfigSetstate = [](nano::tuple const& state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid MropeConfig state!");
        }
        return tle::MropeConfig(nano::cast<tle::Tensor>(state[0]), nano::cast<SizeType32>(state[1]));
    };
    nano::class_<tle::MropeConfig>(m, "MropeConfig")
        .def(nano::init<Tensor, SizeType32>(), nano::arg("mrope_rotary_cos_sin"), nano::arg("mrope_position_deltas"))
        .def_prop_ro("mrope_rotary_cos_sin", &tle::MropeConfig::getMRopeRotaryCosSin)
        .def_prop_ro("mrope_position_deltas", &tle::MropeConfig::getMRopePositionDeltas)
        // .def(nano::pickle(MropeConfigGetstate, MropeConfigSetstate));
        .def("__getstate__", MropeConfigGetstate)
        .def("__setstate__", MropeConfigSetstate);

    auto lookaheadDecodingConfigGetstate = [](tle::LookaheadDecodingConfig const& self)
    { return nano::make_tuple(self.getWindowSize(), self.getNgramSize(), self.getVerificationSetSize()); };
    auto lookaheadDecodingConfigSetstate = [](nano::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid LookaheadDecodingConfig state!");
        }
        return tle::LookaheadDecodingConfig(
            nano::cast<SizeType32>(state[0]), nano::cast<SizeType32>(state[1]), nano::cast<SizeType32>(state[2]));
    };
    nano::class_<tle::LookaheadDecodingConfig>(m, "LookaheadDecodingConfig")
        .def(nano::init<SizeType32, SizeType32, SizeType32>(), nano::arg("max_window_size"),
            nano::arg("max_ngram_size"), nano::arg("max_verification_set_size"))
        .def_prop_ro("max_window_size", &tle::LookaheadDecodingConfig::getWindowSize)
        .def_prop_ro("max_ngram_size", &tle::LookaheadDecodingConfig::getNgramSize)
        .def_prop_ro("max_verification_set_size", &tle::LookaheadDecodingConfig::getVerificationSetSize)
        .def("calculate_speculative_resource", &tle::LookaheadDecodingConfig::calculateSpeculativeResource)
        .def_static(
            "calculate_speculative_resource_tuple", &tle::LookaheadDecodingConfig::calculateSpeculativeResourceTuple)
        // .def(nano::pickle(lookaheadDecodingConfigGetstate, lookaheadDecodingConfigSetstate))
        .def("__getstate__", lookaheadDecodingConfigGetstate)
        .def("__setstate__", lookaheadDecodingConfigSetstate)
        .def_static("get_default_lookahead_decoding_window",
            []() { return tle::LookaheadDecodingConfig::kDefaultLookaheadDecodingWindow; })
        .def_static("get_default_lookahead_decoding_ngram",
            []() { return tle::LookaheadDecodingConfig::kDefaultLookaheadDecodingNgram; })
        .def_static("get_default_lookahead_decoding_verification_set",
            []() { return tle::LookaheadDecodingConfig::kDefaultLookaheadDecodingVerificationSet; });

    auto TokenRangeRetentionConfigGetstate = [](tle::KvCacheRetentionConfig::TokenRangeRetentionConfig const& self)
    { return nano::make_tuple(self.tokenStart, self.tokenEnd, self.priority, self.durationMs); };
    auto TokenRangeRetentionConfigSetstate = [](nano::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::KvCacheRetentionConfig::TokenRangeRetentionConfig(nano::cast<SizeType32>(state[0]),
            nano::cast<std::optional<SizeType32>>(state[1]), nano::cast<tle::RetentionPriority>(state[2]),
            nano::cast<std::optional<std::chrono::milliseconds>>(state[3]));
    };
    auto kvCacheRetentionConfigGetstate = [](tle::KvCacheRetentionConfig const& self)
    {
        return nano::make_tuple(self.getTokenRangeRetentionConfigs(), self.getDecodeRetentionPriority(),
            self.getDecodeDurationMs(), self.getTransferMode(), self.getDirectory());
    };
    auto kvCacheRetentionConfigSetstate = [](nano::tuple const& state)
    {
        if (state.size() != 5)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::KvCacheRetentionConfig(
            nano::cast<std::vector<tle::KvCacheRetentionConfig::TokenRangeRetentionConfig>>(state[0]),
            nano::cast<tle::RetentionPriority>(state[1]),
            nano::cast<std::optional<std::chrono::milliseconds>>(state[2]),
            nano::cast<tle::KvCacheTransferMode>(state[3]), nano::cast<std::optional<std::string>>(state[4]));
    };

    auto kvCacheRetentionConfig = nano::class_<tle::KvCacheRetentionConfig>(m, "KvCacheRetentionConfig");

    nano::class_<tle::KvCacheRetentionConfig::TokenRangeRetentionConfig>(
        kvCacheRetentionConfig, "TokenRangeRetentionConfig")
        .def(nano::init<SizeType32, std::optional<SizeType32>, tle::RetentionPriority,
                 std::optional<std::chrono::milliseconds>>(),
            nano::arg("token_start"), nano::arg("token_end"), nano::arg("priority"),
            nano::arg("duration_ms") = nano::none())
        .def_rw("token_start", &tle::KvCacheRetentionConfig::TokenRangeRetentionConfig::tokenStart)
        .def_rw("token_end", &tle::KvCacheRetentionConfig::TokenRangeRetentionConfig::tokenEnd)
        .def_rw("priority", &tle::KvCacheRetentionConfig::TokenRangeRetentionConfig::priority)
        .def_rw("duration_ms", &tle::KvCacheRetentionConfig::TokenRangeRetentionConfig::durationMs)
        // .def(nano::pickle(TokenRangeRetentionConfigGetstate, TokenRangeRetentionConfigSetstate))
        .def("__getstate__", TokenRangeRetentionConfigGetstate)
        .def("__setstate__", TokenRangeRetentionConfigSetstate)
        .def("__eq__", &tle::KvCacheRetentionConfig::TokenRangeRetentionConfig::operator==);

    // There's a circular dependency between the declaration of the TokenRangeRetentionPriority and
    // KvCacheRetentionConfig bindings. Defer definition of the KvCacheRetentionConfig bindings until the
    // TokenRangeRetentionPriority bindings have been defined.
    kvCacheRetentionConfig
        .def(nano::init<std::vector<tle::KvCacheRetentionConfig::TokenRangeRetentionConfig>, tle::RetentionPriority,
                 std::optional<std::chrono::milliseconds>, tle::KvCacheTransferMode, std::optional<std::string>>(),
            nano::arg("token_range_retention_configs"),
            nano::arg("decode_retention_priority") = tle::KvCacheRetentionConfig::kDefaultRetentionPriority,
            nano::arg("decode_duration_ms") = nano::none(), nano::arg("transfer_mode") = tle::KvCacheTransferMode::DRAM,
            nano::arg("directory") = nano::none())
        .def_prop_ro("token_range_retention_configs", &tle::KvCacheRetentionConfig::getTokenRangeRetentionConfigs)
        .def_prop_ro("decode_retention_priority", &tle::KvCacheRetentionConfig::getDecodeRetentionPriority)
        .def_prop_ro("decode_duration_ms", &tle::KvCacheRetentionConfig::getDecodeDurationMs)
        .def_prop_ro("transfer_mode", &tle::KvCacheRetentionConfig::getTransferMode)
        .def_prop_ro("directory", &tle::KvCacheRetentionConfig::getDirectory)
        // .def(nano::pickle(kvCacheRetentionConfigGetstate, kvCacheRetentionConfigSetstate))
        .def("__getstate__", kvCacheRetentionConfigGetstate)
        .def("__setstate__", kvCacheRetentionConfigSetstate)
        .def("__eq__", &tle::KvCacheRetentionConfig::operator==);

    auto ContextPhaseParamsGetState = [](tle::ContextPhaseParams const& self)
    {
        if (self.getState() != nullptr)
        {
            auto serializedState = self.getSerializedState();
            return nano::make_tuple(self.getFirstGenTokens(), self.getReqId(),
                nano::bytes(serializedState.data(), serializedState.size()), self.getDraftTokens());
        }
        return nano::make_tuple(self.getFirstGenTokens(), self.getReqId(), nano::none(), self.getDraftTokens());
    };

    auto ContextPhaseParamsSetState = [](nano::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid ContextPhaseParams state!");
        }
        if (!state[2].is_none())
        {
            auto opaque_state = nano::cast<nano::bytes>(state[2]);
            auto opaque_state_str_view = std::string_view(nano::cast<std::string_view>(opaque_state));
            return std::make_unique<tle::ContextPhaseParams>(nano::cast<VecTokens>(state[0]),
                nano::cast<tle::ContextPhaseParams::RequestIdType>(state[1]),
                std::vector<char>(opaque_state_str_view.begin(), opaque_state_str_view.end()),
                nano::cast<std::optional<VecTokens>>(state[3]));
        }
        return std::make_unique<tle::ContextPhaseParams>(nano::cast<VecTokens>(state[0]),
            nano::cast<tle::ContextPhaseParams::RequestIdType>(state[1]),
            nano::cast<std::optional<VecTokens>>(state[3]));
    };

    nano::class_<tle::ContextPhaseParams>(m, "ContextPhaseParams")
        .def("__init__",
            [](VecTokens const& first_gen_tokens, tle::ContextPhaseParams::RequestIdType req_id,
                std::optional<nano::bytes> const& opaque_state, std::optional<VecTokens> const& draft_tokens)
            {
                if (opaque_state)
                {
                    auto opaque_state_str_view = nano::cast<std::string_view>(opaque_state.value());
                    return std::make_unique<tle::ContextPhaseParams>(first_gen_tokens, req_id,
                        std::vector<char>(opaque_state_str_view.begin(), opaque_state_str_view.end()), draft_tokens);
                }
                return std::make_unique<tle::ContextPhaseParams>(first_gen_tokens, req_id, draft_tokens);
            })
        // .def_prop_ro("first_gen_tokens", &tle::ContextPhaseParams::getFirstGenTokens)
        // .def_prop_ro("draft_tokens", &tle::ContextPhaseParams::getDraftTokens)
        .def_prop_ro("first_gen_tokens",
            [](tle::ContextPhaseParams const& self) -> decltype(auto) { return self.getFirstGenTokens(); })
        .def_prop_ro(
            "draft_tokens", [](tle::ContextPhaseParams const& self) -> decltype(auto) { return self.getDraftTokens(); })
        .def_prop_ro("req_id", &tle::ContextPhaseParams::getReqId)
        .def_prop_ro("opaque_state",
            [](tle::ContextPhaseParams const& self)
            {
                std::optional<nano::bytes> opaque_state{std::nullopt};
                if (self.getState() != nullptr)
                {
                    auto serializedState = self.getSerializedState();
                    opaque_state = nano::bytes(serializedState.data(), serializedState.size());
                }
                return opaque_state;
            })
        // .def(nano::pickle(ContextPhaseParamsGetState, ContextPhaseParamsSetState));
        .def("__getstate__", ContextPhaseParamsGetState)
        .def("__setstate__", ContextPhaseParamsSetState);

    auto EagleDecodingConfigGetstate = [](tle::EagleConfig const& self)
    {
        return nano::make_tuple(self.getEagleChoices(), self.isGreedySampling(), self.getPosteriorThreshold(),
            self.useDynamicTree(), self.getDynamicTreeMaxTopK());
    };
    auto EagleDecodingConfigSetstate = [](nano::tuple const& state)
    {
        if (state.size() != 5)
        {
            throw std::runtime_error("Invalid EagleConfig state!");
        }
        return tle::EagleConfig(nano::cast<std::optional<tle::EagleChoices>>(state[0]), nano::cast<bool>(state[1]),
            nano::cast<std::optional<float>>(state[2]), nano::cast<bool>(state[3]),
            nano::cast<std::optional<SizeType32>>(state[4]));
    };
    nano::class_<tle::EagleConfig>(m, "EagleConfig")
        .def(
            nano::init<std::optional<tle::EagleChoices>, bool, std::optional<float>, bool, std::optional<SizeType32>>(),
            nano::arg("eagle_choices") = nano::none(), nano::arg("greedy_sampling") = true,
            nano::arg("posterior_threshold") = nano::none(), nano::arg("use_dynamic_tree") = false,
            nano::arg("dynamic_tree_max_topK") = nano::none())
        .def_prop_ro("eagle_choices", &tle::EagleConfig::getEagleChoices)
        .def_prop_ro("greedy_sampling", &tle::EagleConfig::isGreedySampling)
        .def_prop_ro("posterior_threshold", &tle::EagleConfig::getPosteriorThreshold)
        .def_prop_ro("use_dynamic_tree", &tle::EagleConfig::useDynamicTree)
        .def_prop_ro("dynamic_tree_max_topK", &tle::EagleConfig::getDynamicTreeMaxTopK)
        // .def(nano::pickle(EagleDecodingConfigGetstate, EagleDecodingConfigSetstate));
        .def("__getstate__", EagleDecodingConfigGetstate)
        .def("__setstate__", EagleDecodingConfigSetstate);

    // Guided decoding params
    auto pyGuidedDecodingParams = nano::class_<tle::GuidedDecodingParams>(m, "GuidedDecodingParams");

    nano::enum_<tle::GuidedDecodingParams::GuideType>(pyGuidedDecodingParams, "GuideType")
        .value("JSON", tle::GuidedDecodingParams::GuideType::kJSON)
        .value("JSON_SCHEMA", tle::GuidedDecodingParams::GuideType::kJSON_SCHEMA)
        .value("REGEX", tle::GuidedDecodingParams::GuideType::kREGEX)
        .value("EBNF_GRAMMAR", tle::GuidedDecodingParams::GuideType::kEBNF_GRAMMAR)
        .value("STRUCTURAL_TAG", tle::GuidedDecodingParams::GuideType::kSTRUCTURAL_TAG);

    auto guidedDecodingParamsGetstate
        = [](tle::GuidedDecodingParams const& self) { return nano::make_tuple(self.getGuideType(), self.getGuide()); };

    auto guidedDecodingParamsSetstate = [](nano::tuple state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid GuidedDecodingParams state!");
        }
        return tle::GuidedDecodingParams(nano::cast<tle::GuidedDecodingParams::GuideType>(state[0]),
            nano::cast<std::optional<std::string>>(state[1]));
    };

    pyGuidedDecodingParams
        .def(nano::init<tle::GuidedDecodingParams::GuideType, std::optional<std::string>>(), nano::arg("guide_type"),
            nano::arg("guide") = nano::none())
        .def_prop_ro("guide_type", &tle::GuidedDecodingParams::getGuideType)
        .def_prop_ro("guide", &tle::GuidedDecodingParams::getGuide)
        // .def(nano::pickle(guidedDecodingParamsGetstate, guidedDecodingParamsSetstate));
        .def("__getstate__", guidedDecodingParamsGetstate)
        .def("__setstate__", guidedDecodingParamsSetstate);

    auto requestGetstate = [](tle::Request const& self)
    {
        return nano::make_tuple(self.getInputTokenIds(), self.getMaxTokens(), self.getStreaming(),
            self.getSamplingConfig(), self.getOutputConfig(), self.getEndId(), self.getPadId(), self.getPositionIds(),
            self.getBadWords(), self.getStopWords(), self.getEmbeddingBias(), self.getExternalDraftTokensConfig(),
            self.getPromptTuningConfig(), self.getMultimodalEmbedding(), self.getMropeConfig(), self.getLoraConfig(),
            self.getLookaheadConfig(), self.getKvCacheRetentionConfig(), self.getLogitsPostProcessorName(),
            self.getLogitsPostProcessor(), self.getEncoderInputTokenIds(), self.getClientId(),
            self.getReturnAllGeneratedTokens(), self.getPriority(), self.getRequestType(), self.getContextPhaseParams(),
            self.getEncoderInputFeatures(), self.getEncoderOutputLength(), self.getCrossAttentionMask(),
            self.getEagleConfig(), self.getSkipCrossAttnBlocks(), self.getGuidedDecodingParams());
    };
    auto requestSetstate = [](nano::tuple const& state)
    {
        if (state.size() != 32)
        {
            throw std::runtime_error("Invalid Request state!");
        }
        return std::make_unique<tle::Request>(nano::cast<VecTokens>(state[0]), nano::cast<SizeType32>(state[1]),
            nano::cast<bool>(state[2]), nano::cast<tle::SamplingConfig>(state[3]),
            nano::cast<tle::OutputConfig>(state[4]), nano::cast<std::optional<SizeType32>>(state[5]),
            nano::cast<std::optional<SizeType32>>(state[6]),
            nano::cast<std::optional<std::vector<SizeType32>>>(state[7]),
            nano::cast<std::optional<std::list<VecTokens>>>(state[8]),
            nano::cast<std::optional<std::list<VecTokens>>>(state[9]), nano::cast<std::optional<Tensor>>(state[10]),
            nano::cast<std::optional<tle::ExternalDraftTokensConfig>>(state[11]),
            nano::cast<std::optional<tle::PromptTuningConfig>>(state[12]), nano::cast<std::optional<Tensor>>(state[13]),
            nano::cast<std::optional<tle::MropeConfig>>(state[14]),
            nano::cast<std::optional<tle::LoraConfig>>(state[15]),
            nano::cast<std::optional<tle::LookaheadDecodingConfig>>(state[16]),
            nano::cast<std::optional<tle::KvCacheRetentionConfig>>(state[17]),
            nano::cast<std::optional<std::string>>(state[18]),
            nano::cast<std::optional<tle::LogitsPostProcessor>>(state[19]),
            nano::cast<std::optional<VecTokens>>(state[20]), nano::cast<std::optional<IdType>>(state[21]),
            nano::cast<bool>(state[22]), nano::cast<tle::PriorityType>(state[23]),
            nano::cast<tle::RequestType>(state[24]), nano::cast<std::optional<tle::ContextPhaseParams>>(state[25]),
            nano::cast<std::optional<tle::Tensor>>(state[26]), nano::cast<std::optional<SizeType32>>(state[27]),
            nano::cast<std::optional<tle::Tensor>>(state[28]), 1,
            nano::cast<std::optional<tle::EagleConfig>>(state[29]), nano::cast<std::optional<tle::Tensor>>(state[30]),
            nano::cast<std::optional<tle::GuidedDecodingParams>>(state[31]));
    };

    nano::class_<tle::Request> request(m, "Request", nano::dynamic_attr());
    request
        .def(nano::init<tle::VecTokens,                         // inputTokenIds
                 tle::SizeType32,                               // maxTokens
                 bool,                                          // streaming
                 tle::SamplingConfig const&,                    // samplingConfig
                 tle::OutputConfig const&,                      // outputConfig
                 std::optional<tle::SizeType32> const&,         // endId
                 std::optional<tle::SizeType32> const&,         // padId
                 std::optional<std::vector<SizeType32>>,        // positionIds
                 std::optional<std::list<tle::VecTokens>>,      // badWords
                 std::optional<std::list<tle::VecTokens>>,      // stopWords
                 std::optional<tle::Tensor>,                    // embeddingBias
                 std::optional<tle::ExternalDraftTokensConfig>, // externalDraftTokensConfig
                 std::optional<tle::PromptTuningConfig>,        // pTuningConfig
                 std::optional<tle::Tensor>,                    // multimodalEmbedding
                 std::optional<tle::MropeConfig>,               // mRopeConfig
                 std::optional<tle::LoraConfig>,                // loraConfig
                 std::optional<tle::LookaheadDecodingConfig>,   // lookaheadConfig
                 std::optional<tle::KvCacheRetentionConfig>,    // kvCacheRetentionConfig
                 std::optional<std::string>,                    // logitsPostProcessorName
                 std::optional<tle::LogitsPostProcessor>,       // logitsPostProcessor
                 std::optional<tle::VecTokens>,                 // encoderInputTokenIds
                 std::optional<tle::IdType>,                    // clientId
                 bool,                                          // returnAllGeneratedTokens
                 tle::PriorityType,                             // priority
                 tle::RequestType,                              // type
                 std::optional<tle::ContextPhaseParams>,        // contextPhaseParams
                 std::optional<tle::Tensor>,                    // encoderInputFeatures
                 std::optional<tle::SizeType32>,                // encoderOutputLength
                 std::optional<tle::Tensor>,                    // crossAttentionMask
                 SizeType32,                                    // numReturnSequences
                 std::optional<tle::EagleConfig>,               // eagleConfig
                 std::optional<tle::Tensor>,                    // skipCrossAttnBlocks
                 std::optional<tle::GuidedDecodingParams>,      // guidedDecodingParams
                 std::optional<tle::SizeType32>,                // languageAdapterUid
                 std::optional<tle::MillisecondsType>           // allottedTimeMs
                 >(),
            // clang-format off
        nano::arg("input_token_ids"),
        nano::arg("max_tokens"),
        nano::kw_only(),
        nano::arg("streaming") = false,
        nano::arg("sampling_config") = tle::SamplingConfig(),
        nano::arg("output_config") = tle::OutputConfig(),
        nano::arg("end_id") = nano::none(),
        nano::arg("pad_id") = nano::none(),
        nano::arg("position_ids") = nano::none(),
        nano::arg("bad_words") = nano::none(),
        nano::arg("stop_words") = nano::none(),
        nano::arg("embedding_bias") = nano::none(),
        nano::arg("external_draft_tokens_config") = nano::none(),
        nano::arg("prompt_tuning_config") = nano::none(),
        nano::arg("multimodal_embedding") = nano::none(),
        nano::arg("mrope_config") = nano::none(),
        nano::arg("lora_config") = nano::none(),
        nano::arg("lookahead_config") = nano::none(),
        nano::arg("kv_cache_retention_config") = nano::none(),
        nano::arg("logits_post_processor_name") = nano::none(),
        nano::arg("logits_post_processor") = nano::none(),
        nano::arg("encoder_input_token_ids") = nano::none(),
        nano::arg("client_id") = nano::none(),
        nano::arg("return_all_generated_tokens") = false,
        nano::arg("priority") = tle::Request::kDefaultPriority,
        nano::arg("type") = tle::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION,
        nano::arg("context_phase_params") = nano::none(),
        nano::arg("encoder_input_features") = nano::none(),
        nano::arg("encoder_output_length") = nano::none(),
        nano::arg("cross_attention_mask") = nano::none(),
        nano::arg("num_return_sequences") = 1,
        nano::arg("eagle_config") = nano::none(),
        nano::arg("skip_cross_attn_blocks") = nano::none(),
        nano::arg("guided_decoding_params") = nano::none(),
        nano::arg("language_adapter_uid") = nano::none(),
        nano::arg("allotted_time_ms") = nano::none()
    )      // clang-format on
        .def_prop_ro("input_token_ids", &tle::Request::getInputTokenIds)
        .def_prop_ro("max_tokens", &tle::Request::getMaxTokens)
        .def_prop_rw("streaming", &tle::Request::getStreaming, &tle::Request::setStreaming)
        .def_prop_rw("sampling_config", &tle::Request::getSamplingConfig, &tle::Request::setSamplingConfig)
        .def_prop_rw("output_config", &tle::Request::getOutputConfig, &tle::Request::setOutputConfig)
        .def_prop_rw("end_id", &tle::Request::getEndId, &tle::Request::setEndId)
        .def_prop_rw("pad_id", &tle::Request::getPadId, &tle::Request::setPadId)
        .def_prop_rw("position_ids", &tle::Request::getPositionIds, &tle::Request::setPositionIds)
        .def_prop_rw("bad_words", &tle::Request::getBadWords, &tle::Request::setBadWords)
        .def_prop_rw("stop_words", &tle::Request::getStopWords, &tle::Request::setStopWords)
        .def_prop_rw("embedding_bias", &tle::Request::getEmbeddingBias, &tle::Request::setEmbeddingBias)
        .def_prop_rw("external_draft_tokens_config", &tle::Request::getExternalDraftTokensConfig,
            &tle::Request::setExternalDraftTokensConfig)
        .def_prop_rw("prompt_tuning_config", &tle::Request::getPromptTuningConfig, &tle::Request::setPromptTuningConfig)
        .def_prop_rw(
            "multimodal_embedding", &tle::Request::getMultimodalEmbedding, &tle::Request::setMultimodalEmbedding)
        .def_prop_rw("mrope_config", &tle::Request::getMropeConfig, &tle::Request::setMropeConfig)
        .def_prop_rw("lora_config", &tle::Request::getLoraConfig, &tle::Request::setLoraConfig)
        .def_prop_rw("lookahead_config", &tle::Request::getLookaheadConfig, &tle::Request::setLookaheadConfig)
        .def_prop_rw("kv_cache_retention_config", &tle::Request::getKvCacheRetentionConfig,
            &tle::Request::setKvCacheRetentionConfig)
        .def_prop_rw("logits_post_processor_name", &tle::Request::getLogitsPostProcessorName,
            &tle::Request::setLogitsPostProcessorName)
        .def_prop_rw(
            "logits_post_processor", &tle::Request::getLogitsPostProcessor, &tle::Request::setLogitsPostProcessor)
        .def_prop_rw(
            "encoder_input_token_ids", &tle::Request::getEncoderInputTokenIds, &tle::Request::setEncoderInputTokenIds)
        .def_prop_rw("client_id", &tle::Request::getClientId, &tle::Request::setClientId)
        .def_prop_rw("return_all_generated_tokens", &tle::Request::getReturnAllGeneratedTokens,
            &tle::Request::setReturnAllGeneratedTokens)
        .def_prop_rw("request_type", &tle::Request::getRequestType, &tle::Request::setRequestType)
        .def_prop_rw(
            "encoder_input_features", &tle::Request::getEncoderInputFeatures, &tle::Request::setEncoderInputFeatures)
        .def_prop_rw("cross_attention_mask", &tle::Request::getCrossAttentionMask, &tle::Request::setCrossAttentionMask)
        .def_prop_rw("eagle_config", &tle::Request::getEagleConfig, &tle::Request::setEagleConfig)
        .def_prop_rw(
            "skip_cross_attn_blocks", &tle::Request::getSkipCrossAttnBlocks, &tle::Request::setSkipCrossAttnBlocks)
        .def_prop_rw(
            "guided_decoding_params", &tle::Request::getGuidedDecodingParams, &tle::Request::setGuidedDecodingParams)
        .def_prop_rw("allotted_time_ms", &tle::Request::getAllottedTimeMs, &tle::Request::setAllottedTimeMs)
        .def_prop_rw("context_phase_params", &tle::Request::getContextPhaseParams, &tle::Request::setContextPhaseParams)
        // .def(nano::pickle(requestGetstate, requestSetstate));
        .def("__getstate__", requestGetstate)
        .def("__setstate__", requestSetstate);
    request.attr("BATCHED_POST_PROCESSOR_NAME") = tle::Request::kBatchedPostProcessorName;

    nano::class_<tle::SpeculativeDecodingFastLogitsInfo>(m, "SpeculativeDecodingFastLogitsInfo")
        .def(nano::init<>())
        .def_rw("draft_request_id", &tle::SpeculativeDecodingFastLogitsInfo::draftRequestId)
        .def_rw("draft_participant_id", &tle::SpeculativeDecodingFastLogitsInfo::draftParticipantId)
        .def("to_tensor", &tle::SpeculativeDecodingFastLogitsInfo::toTensor);

    auto requestPerfMetrics = nano::class_<tle::RequestPerfMetrics>(m, "RequestPerfMetrics");

    nano::class_<tle::RequestPerfMetrics::TimingMetrics>(m, "TimingMetrics")
        .def(nano::init<>())
        .def_rw("arrival_time", &tle::RequestPerfMetrics::TimingMetrics::arrivalTime)
        .def_rw("first_scheduled_time", &tle::RequestPerfMetrics::TimingMetrics::firstScheduledTime)
        .def_rw("first_token_time", &tle::RequestPerfMetrics::TimingMetrics::firstTokenTime)
        .def_rw("last_token_time", &tle::RequestPerfMetrics::TimingMetrics::lastTokenTime)
        .def_rw("kv_cache_transfer_start", &tle::RequestPerfMetrics::TimingMetrics::kvCacheTransferStart)
        .def_rw("kv_cache_transfer_end", &tle::RequestPerfMetrics::TimingMetrics::kvCacheTransferEnd);

    nano::class_<tle::RequestPerfMetrics::KvCacheMetrics>(m, "KvCacheMetrics")
        .def(nano::init<>())
        .def_rw("num_total_allocated_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numTotalAllocatedBlocks)
        .def_rw("num_new_allocated_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numNewAllocatedBlocks)
        .def_rw("num_reused_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numReusedBlocks)
        .def_rw("num_missed_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numMissedBlocks)
        .def_rw("kv_cache_hit_rate", &tle::RequestPerfMetrics::KvCacheMetrics::kvCacheHitRate);

    nano::class_<tle::RequestPerfMetrics::SpeculativeDecodingMetrics>(m, "SpeculativeDecodingMetrics")
        .def(nano::init<>())
        .def_rw("acceptance_rate", &tle::RequestPerfMetrics::SpeculativeDecodingMetrics::acceptanceRate)
        .def_rw("total_accepted_draft_tokens",
            &tle::RequestPerfMetrics::SpeculativeDecodingMetrics::totalAcceptedDraftTokens)
        .def_rw("total_draft_tokens", &tle::RequestPerfMetrics::SpeculativeDecodingMetrics::totalDraftTokens);

    // There's a circular dependency between the declaration of the TimingMetrics and RequestPerfMetrics bindings.
    // Defer definition of the RequestPerfMetrics bindings until the TimingMetrics have been defined.
    requestPerfMetrics.def(nano::init<>())
        .def_rw("timing_metrics", &tle::RequestPerfMetrics::timingMetrics)
        .def_rw("kv_cache_metrics", &tle::RequestPerfMetrics::kvCacheMetrics)
        .def_rw("speculative_decoding", &tle::RequestPerfMetrics::speculativeDecoding)
        .def_rw("first_iter", &tle::RequestPerfMetrics::firstIter)
        .def_rw("last_iter", &tle::RequestPerfMetrics::lastIter)
        .def_rw("iter", &tle::RequestPerfMetrics::iter);

    nano::class_<tle::AdditionalOutput>(m, "AdditionalOutput")
        .def("__init__",
            [](std::string const& name, tle::Tensor const& output)
            { return std::make_unique<tle::AdditionalOutput>(name, output); })
        .def_rw("name", &tle::AdditionalOutput::name)
        .def_rw("output", &tle::AdditionalOutput::output);

    auto resultSetstate = [](nano::tuple const& state)
    {
        if (state.size() != 12)
        {
            throw std::runtime_error("Invalid Request state!");
        }
        tle::Result result;
        result.isFinal = nano::cast<bool>(state[0]);
        result.outputTokenIds = nano::cast<std::vector<VecTokens>>(state[1]);
        result.cumLogProbs = nano::cast<std::optional<std::vector<float>>>(state[2]);
        result.logProbs = nano::cast<std::optional<std::vector<std::vector<float>>>>(state[3]);
        result.contextLogits = nano::cast<std::optional<Tensor>>(state[4]);
        result.generationLogits = nano::cast<std::optional<Tensor>>(state[5]);
        result.encoderOutput = nano::cast<std::optional<Tensor>>(state[6]);
        result.finishReasons = nano::cast<std::vector<tle::FinishReason>>(state[7]);
        result.sequenceIndex = nano::cast<SizeType32>(state[8]);
        result.isSequenceFinal = nano::cast<bool>(state[9]);
        result.decodingIter = nano::cast<SizeType32>(state[10]);
        result.contextPhaseParams = nano::cast<std::optional<tle::ContextPhaseParams>>(state[11]);
        return std::make_unique<tle::Result>(result);
    };

    auto resultGetstate = [](tle::Result const& self)
    {
        return nano::make_tuple(self.isFinal, self.outputTokenIds, self.cumLogProbs, self.logProbs, self.contextLogits,
            self.generationLogits, self.encoderOutput, self.finishReasons, self.sequenceIndex, self.isSequenceFinal,
            self.decodingIter, self.contextPhaseParams);
    };

    nano::class_<tle::Result>(m, "Result")
        .def(nano::init<>())
        .def_rw("is_final", &tle::Result::isFinal)
        .def_rw("output_token_ids", &tle::Result::outputTokenIds)
        .def_rw("cum_log_probs", &tle::Result::cumLogProbs)
        .def_rw("log_probs", &tle::Result::logProbs)
        .def_rw("context_logits", &tle::Result::contextLogits)
        .def_rw("generation_logits", &tle::Result::generationLogits)
        .def_rw("spec_dec_fast_logits_info", &tle::Result::specDecFastLogitsInfo)
        .def_rw("encoder_output", &tle::Result::encoderOutput)
        .def_rw("finish_reasons", &tle::Result::finishReasons)
        .def_rw("sequence_index", &tle::Result::sequenceIndex)
        .def_rw("is_sequence_final", &tle::Result::isSequenceFinal)
        .def_rw("decoding_iter", &tle::Result::decodingIter)
        .def_rw("context_phase_params", &tle::Result::contextPhaseParams)
        .def_rw("request_perf_metrics", &tle::Result::requestPerfMetrics)
        .def_rw("additional_outputs", &tle::Result::additionalOutputs)
        .def_rw("context_phase_params", &tle::Result::contextPhaseParams)
        // .def(nano::pickle(resultGetstate, resultSetstate));
        .def("__getstate__", resultGetstate)
        .def("__setstate__", resultSetstate);

    auto responseGetstate = [](tle::Response const& self)
    {
        NVTX3_SCOPED_RANGE(responseGetstate);
        return nano::make_tuple(self.getRequestId(), self.getResult(), self.getClientId());
    };

    auto responseSetstate = [](nano::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid Request state!");
        }
        return std::make_unique<tle::Response>(
            nano::cast<SizeType32>(state[0]), nano::cast<tle::Result>(state[1]), nano::cast<SizeType32>(state[2]));
    };

    nano::class_<tle::Response>(m, "Response")
        .def(nano::init<IdType, std::string, std::optional<IdType>>(), nano::arg("request_id"), nano::arg("error_msg"),
            nano::arg("client_id") = std::nullopt)
        .def(nano::init<IdType, tle::Result, std::optional<IdType>>(), nano::arg("request_id"), nano::arg("result"),
            nano::arg("client_id") = std::nullopt)
        .def_prop_ro("request_id", &tle::Response::getRequestId)
        .def_prop_ro("client_id", &tle::Response::getClientId)
        .def("has_error", &tle::Response::hasError)
        .def_prop_ro("error_msg", &tle::Response::getErrorMsg)
        .def_prop_ro("result", &tle::Response::getResult)
        .def("clear_context_logits",
            [](tle::Response& self)
            {
                if (!self.hasError())
                {
                    auto& result = const_cast<tle::Result&>(self.getResult());
                    result.contextLogits.reset();
                }
            })
        .def("clear_generation_logits",
            [](tle::Response& self)
            {
                if (!self.hasError())
                {
                    auto& result = const_cast<tle::Result&>(self.getResult());
                    result.generationLogits.reset();
                }
            })
        // .def(nano::pickle(responseGetstate, responseSetstate));
        .def("__getstate__", responseGetstate)
        .def("__setstate__", responseSetstate);

    m.def(
        "serialize_responses",
        [](std::vector<tle::Response> const& responses)
        {
            nano::gil_scoped_release release;
            NVTX3_SCOPED_RANGE(serialize_responses);
            return tle::Serialization::serialize(responses);
        },
        nano::arg("serialize_responses"), "Serializes a list of Response objects.");

    m.def(
        "deserialize_responses",
        [](std::vector<char>& data)
        {
            nano::gil_scoped_release release;
            NVTX3_SCOPED_RANGE(deserialize_responses);
            return tle::Serialization::deserializeResponses(data);
        },
        nano::arg("deserialize_responses"), "Deserializes a list into Response objects.");
}

} // namespace tensorrt_llm::nanobind::executor
