
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "engine/palisade_context.h"
#include "hebench/api_bridge/cpp/hebench.hpp"

class PalisadeEngine;

namespace pbe {
namespace ckks {

class MatMultCipherBatchAxisBenchmarkDescription : public hebench::cpp::BenchmarkDescription
{
public:
    HEBERROR_DECLARE_CLASS_NAME(MatMultCipherBatchAxisBenchmarkDescription)

public:
    static constexpr std::int64_t MatMultCipherBatchAxisID = 2;
    static constexpr std::uint64_t OpParametersCount       = 2; // number of parameters for this operation
    static constexpr std::uint64_t ResultComponentsCount   = 1; // number of components of result for this operation

    static constexpr const char *AlgorithmName        = "MatMultCipherBatchAxis";
    static constexpr const char *AlgorithmDescription = "One matrix element per ciphertext";

    // HE specific parameters
    static constexpr std::size_t DefaultPolyModulusDegree    = 16384; // 8192 doesn't work because of PALISADE hybrid mode
    static constexpr std::size_t DefaultNumCoefficientModuli = 2;
    static constexpr int DefaultScaleExponent                = 40;

    enum : std::uint64_t
    {
        Index_WParamsStart = 0,
        Index_rows_M0      = Index_WParamsStart,
        Index_cols_M0,
        Index_cols_M1,
        Index_ExtraWParamsStart,
        Index_PolyModulusDegree = Index_ExtraWParamsStart,
        Index_NumCoefficientModuli,
        Index_ScaleExponent,
        NumWorkloadParams // This workload requires 3 parameters, and we add 3 encryption params
    };

    MatMultCipherBatchAxisBenchmarkDescription();
    ~MatMultCipherBatchAxisBenchmarkDescription() override;

    std::string getBenchmarkDescription(const hebench::APIBridge::WorkloadParams *p_w_params) const override;

    hebench::cpp::BaseBenchmark *createBenchmark(hebench::cpp::BaseEngine &engine,
                                                 const hebench::APIBridge::WorkloadParams *p_params) override;
    void destroyBenchmark(hebench::cpp::BaseBenchmark *p_bench) override;
};

class MatMultCipherBatchAxisBenchmark : public hebench::cpp::BaseBenchmark
{
public:
    HEBERROR_DECLARE_CLASS_NAME(MatMultCipherBatchAxisBenchmark)

public:
    static constexpr std::int64_t tag = 0x20 + MatMultCipherBatchAxisBenchmarkDescription::MatMultCipherBatchAxisID;

    MatMultCipherBatchAxisBenchmark(PalisadeEngine &engine,
                                    const hebench::APIBridge::BenchmarkDescriptor &bench_desc,
                                    const hebench::APIBridge::WorkloadParams &bench_params);
    ~MatMultCipherBatchAxisBenchmark() override;

    hebench::APIBridge::Handle encode(const hebench::APIBridge::PackedData *p_parameters) override;
    void decode(hebench::APIBridge::Handle encoded_data, hebench::APIBridge::PackedData *p_native) override;
    hebench::APIBridge::Handle encrypt(hebench::APIBridge::Handle encoded_data) override;
    hebench::APIBridge::Handle decrypt(hebench::APIBridge::Handle encrypted_data) override;

    hebench::APIBridge::Handle load(const hebench::APIBridge::Handle *p_local_data, std::uint64_t count) override;
    void store(hebench::APIBridge::Handle remote_data,
               hebench::APIBridge::Handle *p_local_data, std::uint64_t count) override;

    hebench::APIBridge::Handle operate(hebench::APIBridge::Handle h_remote_packed,
                                       const hebench::APIBridge::ParameterIndexer *p_param_indexers) override;

    std::int64_t classTag() const override { return BaseBenchmark::classTag() | MatMultCipherBatchAxisBenchmark::tag; }

private:
    struct MatrixDims
    {
        std::uint64_t rows;
        std::uint64_t cols;
    };
    template <class T>
    /**
     * @brief Encapsulates a column major matrix.
     * @details Each operation parameter is encoded one plaintext/ciphertext
     * per matrix element in column major format.
     */
    class OpParamSample
    {
    public:
        OpParamSample(std::uint64_t rows, std::uint64_t cols) :
            m_rows(rows),
            m_cols(cols),
            m_data(std::make_shared<std::vector<T>>(rows * cols))
        {
        }
        OpParamSample(const OpParamSample &) = default;
        OpParamSample(OpParamSample &&)      = default;
        OpParamSample &operator=(const OpParamSample &) = default;
        OpParamSample &operator=(OpParamSample &&) = default;
        T &at(std::uint64_t row, std::uint64_t col)
        {
            std::uint64_t idx = col * rows() + row;
            return m_data->at(idx);
        }
        const T &at(std::uint64_t row, std::uint64_t col) const
        {
            std::uint64_t idx = col * rows() + row;
            return m_data->at(idx);
        }
        T &at(std::size_t index) { return m_data->at(index); }
        const T &at(std::size_t index) const { return m_data->at(index); }
        std::size_t size() const { return m_data->size(); }
        std::uint64_t rows() const { return m_rows; }
        std::uint64_t cols() const { return m_cols; }

    private:
        std::uint64_t m_rows;
        std::uint64_t m_cols;
        std::shared_ptr<std::vector<T>> m_data;
    };
    typedef OpParamSample<lbcrypto::Plaintext> OpParamSamplePlain;
    typedef OpParamSample<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> OpParamSampleCipher;

    PalisadeContext::Ptr m_p_context;
    hebench::cpp::WorkloadParams::MatrixMultiply m_w_params;
};

} // namespace ckks
} // namespace pbe
