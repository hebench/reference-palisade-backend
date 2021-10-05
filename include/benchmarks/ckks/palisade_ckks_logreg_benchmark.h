
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "engine/palisade_context.h"
#include "hebench/api_bridge/cpp/hebench.hpp"

class PalisadeEngine;

namespace pbe {
namespace ckks {

class LogRegBenchmarkDescription : public hebench::cpp::BenchmarkDescription
{
public:
    HEBERROR_DECLARE_CLASS_NAME(LogRegBenchmarkDescription)

public:
    static constexpr std::uint64_t OpParamsCount    = 3; // number of operation parameters (W, b, X)
    static constexpr std::uint64_t DefaultBatchSize = 100;
    static constexpr std::int64_t LogRegOtherID     = 0;

    enum : std::size_t
    {
        Index_W = 0,
        Index_b,
        Index_X
    };

    static constexpr const char *AlgorithmName        = "EvalPoly";
    static constexpr const char *AlgorithmDescription = "using PALISADE EvalPoly";

    // HE specific parameters
    static constexpr std::size_t DefaultPolyModulusDegree    = 16384; // 8192 doesn't work because of PALISADE hybrid mode
    static constexpr std::size_t DefaultNumCoefficientModuli = 5;
    static constexpr int DefaultScaleExponent                = 45;

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

    LogRegBenchmarkDescription(hebench::APIBridge::Category category, std::size_t batch_size = 0);
    ~LogRegBenchmarkDescription() override;

    std::string getBenchmarkDescription(const hebench::APIBridge::WorkloadParams *p_w_params) const override;

    hebench::cpp::BaseBenchmark *createBenchmark(hebench::cpp::BaseEngine &engine,
                                                 const hebench::APIBridge::WorkloadParams *p_params) override;
    void destroyBenchmark(hebench::cpp::BaseBenchmark *p_bench) override;
};

class LogRegBenchmark : public hebench::cpp::BaseBenchmark
{
public:
    HEBERROR_DECLARE_CLASS_NAME(LogRegBenchmark)

public:
    static constexpr std::int64_t tag = 0x20 + LogRegBenchmarkDescription::LogRegOtherID;

    LogRegBenchmark(PalisadeEngine &engine,
                    const hebench::APIBridge::BenchmarkDescriptor &bench_desc,
                    const hebench::APIBridge::WorkloadParams &bench_params);
    ~LogRegBenchmark() override;

    hebench::APIBridge::Handle encode(const hebench::APIBridge::PackedData *p_parameters) override;
    void decode(hebench::APIBridge::Handle encoded_data, hebench::APIBridge::PackedData *p_native) override;
    hebench::APIBridge::Handle encrypt(hebench::APIBridge::Handle encoded_data) override;
    hebench::APIBridge::Handle decrypt(hebench::APIBridge::Handle encrypted_data) override;

    hebench::APIBridge::Handle load(const hebench::APIBridge::Handle *p_local_data, std::uint64_t count) override;
    void store(hebench::APIBridge::Handle remote_data,
               hebench::APIBridge::Handle *p_local_data, std::uint64_t count) override;

    hebench::APIBridge::Handle operate(hebench::APIBridge::Handle h_remote_packed,
                                       const hebench::APIBridge::ParameterIndexer *p_param_indexers) override;

    std::int64_t classTag() const override { return BaseBenchmark::classTag() | LogRegBenchmark::tag; }

private:
    typedef std::tuple<lbcrypto::Plaintext, lbcrypto::Plaintext, std::vector<std::vector<lbcrypto::Plaintext>>> EncodedOpParams;
    typedef std::tuple<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>, std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>> EncryptedOpParams;

    static constexpr std::int64_t tagEncodedOpParams   = 0x10;
    static constexpr std::int64_t tagEncryptedOpParams = 0x20;
    static constexpr std::int64_t tagEncryptedResult   = 0x40;
    static constexpr std::int64_t tagEncodedResult     = 0x80;
    // coefficients for sigmoid polynomial approx
    static const std::vector<double> SigmoidPolyCoeff;

    static std::vector<std::vector<double>> prepareData(const hebench::APIBridge::DataPack &data_pack);
    std::vector<std::vector<double>> prepareInputs(const hebench::APIBridge::DataPack &data_pack);
    lbcrypto::Plaintext encodeWeightsFromDataPack(const hebench::APIBridge::DataPack &data_pack);
    lbcrypto::Plaintext encodeBiasFromDataPack(const hebench::APIBridge::DataPack &data_pack);
    std::vector<std::vector<lbcrypto::Plaintext>> encodeInputsFromDataPack(const hebench::APIBridge::DataPack &data_pack);
    lbcrypto::Plaintext encodeVector(const std::vector<double> &data);
    lbcrypto::Plaintext encodeBias(double data);
    /**
     * @brief encodeInputs
     * @param[in] data Input data. One input sample per element: [ input_sample_count , n: number_of_features ]
     * @return Batches of encoded input: retval[ batch_count > 0 , input_batch_size <= pmd / 2 ]
     */
    std::vector<std::vector<lbcrypto::Plaintext>> encodeInputs(const std::vector<std::vector<double>> &data);
    std::vector<double> decodeResult(const std::vector<lbcrypto::Plaintext> &pt_result, std::size_t n_samples) const;
    std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> encryptInputs(const std::vector<std::vector<lbcrypto::Plaintext>> &encoded_inputs);
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
    doLogReg(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &w,
             const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &bias,
             const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &inputs,
             std::size_t weights_count,
             int threads = 1);

    std::uint64_t m_sample_size;
    PalisadeContext::Ptr m_p_context;
    hebench::cpp::WorkloadParams::LogisticRegression m_w_params;
};

} // namespace ckks
} // namespace pbe
