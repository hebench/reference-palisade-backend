
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "engine/palisade_context.h"
#include "hebench/api_bridge/cpp/hebench.hpp"

class PalisadeEngine;

namespace pbe {
namespace ckks {

class EltwiseAddPlainCipherBenchmarkDescription : public hebench::cpp::BenchmarkDescription
{
public:
    HEBERROR_DECLARE_CLASS_NAME(EltwiseAddPlainCipherBenchmarkDescription)

public:
    static constexpr std::uint64_t NumOpParams           = 2; // number of operands for this workload
    static constexpr std::uint64_t ResultComponentsCount = 1; // number of components of result for this operation

    // Flexible workload params

    // HE specific parameters
    static constexpr std::size_t DefaultPolyModulusDegree    = 8192;
    static constexpr std::size_t DefaultNumCoefficientModuli = 1;
    static constexpr int DefaultScaleExponent                = 40;

    // This workload (EltwiseAdd) requires only 1 flexible parameters, and we add 3 encryption params
    enum : std::uint64_t
    {
        Index_WParamsStart = 0,
        Index_n            = Index_WParamsStart,
        Index_ExtraWParamsStart,
        Index_PolyModulusDegree = Index_ExtraWParamsStart,
        Index_NumCoefficientModuli,
        Index_ScaleExponent,
        NumWorkloadParams
    };

    EltwiseAddPlainCipherBenchmarkDescription();
    ~EltwiseAddPlainCipherBenchmarkDescription() override;

    std::string getBenchmarkDescription(const hebench::APIBridge::WorkloadParams *p_w_params) const override;

    hebench::cpp::BaseBenchmark *createBenchmark(hebench::cpp::BaseEngine &engine,
                                                 const hebench::APIBridge::WorkloadParams *p_params) override;
    void destroyBenchmark(hebench::cpp::BaseBenchmark *p_bench) override;
};

class EltwiseAddPlainCipherBenchmark : public hebench::cpp::BaseBenchmark
{
public:
    HEBERROR_DECLARE_CLASS_NAME(EltwiseAddPlainCipherBenchmark)

public:
    static constexpr std::int64_t tag = 0x1;

    EltwiseAddPlainCipherBenchmark(PalisadeEngine &engine,
                                   const hebench::APIBridge::BenchmarkDescriptor &bench_desc,
                                   const hebench::APIBridge::WorkloadParams &bench_params);
    ~EltwiseAddPlainCipherBenchmark() override;

    hebench::APIBridge::Handle encode(const hebench::APIBridge::DataPackCollection *p_parameters) override;
    void decode(hebench::APIBridge::Handle encoded_data, hebench::APIBridge::DataPackCollection *p_native) override;
    hebench::APIBridge::Handle encrypt(hebench::APIBridge::Handle encoded_data) override;
    hebench::APIBridge::Handle decrypt(hebench::APIBridge::Handle encrypted_data) override;

    hebench::APIBridge::Handle load(const hebench::APIBridge::Handle *p_local_data, std::uint64_t count) override;
    void store(hebench::APIBridge::Handle remote_data,
               hebench::APIBridge::Handle *p_local_data, std::uint64_t count) override;

    hebench::APIBridge::Handle operate(hebench::APIBridge::Handle h_remote_packed,
                                       const hebench::APIBridge::ParameterIndexer *p_param_indexers,
                                       std::uint64_t indexers_count) override;

    std::int64_t classTag() const override { return BaseBenchmark::classTag() | EltwiseAddPlainCipherBenchmark::tag; }

private:
    // used to bundle a collection of operation parameters
    struct InternalParams
    {
    public:
        static constexpr std::int64_t tagPlaintext  = 0x10;
        static constexpr std::int64_t tagCiphertext = 0x20;

        std::vector<std::shared_ptr<void>> samples;
        std::int64_t tag;
        std::uint64_t param_position;
    };

    PalisadeContext::Ptr m_p_context;
};

} // namespace ckks
} // namespace pbe
