
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "engine/palisade_context.h"
#include "hebench/api_bridge/cpp/hebench.hpp"

class PalisadeEngine;

namespace pbe {
namespace ckks {

class MatMultEIPBenchmarkDescription : public hebench::cpp::BenchmarkDescription
{
public:
    HEBERROR_DECLARE_CLASS_NAME(MatMultEIPBenchmarkDescription)

public:
    static constexpr std::int64_t MatMultEIPID = 1;

    static constexpr const char *AlgorithmName        = "MatMultEIP";
    static constexpr const char *AlgorithmDescription = "One matrix row per ciphertext, using PALISADE EIP";

    // HE specific parameters
    static constexpr std::size_t DefaultPolyModulusDegree    = 8192;
    static constexpr std::size_t DefaultNumCoefficientModuli = 1;
    static constexpr int DefaultScaleExponent                = 40;

    // other workload parameters
    static constexpr std::size_t DefaultNumThreads = 0; // 0 - use all available threads

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
        Index_NumThreads,
        NumWorkloadParams // This workload requires 3 parameters, and we add 3 encryption params
    };

    MatMultEIPBenchmarkDescription();
    ~MatMultEIPBenchmarkDescription() override;

    std::string getBenchmarkDescription(const hebench::APIBridge::WorkloadParams *p_w_params) const override;

    hebench::cpp::BaseBenchmark *createBenchmark(hebench::cpp::BaseEngine &engine,
                                                 const hebench::APIBridge::WorkloadParams *p_params) override;
    void destroyBenchmark(hebench::cpp::BaseBenchmark *p_bench) override;
};

class MatMultEIPBenchmark : public hebench::cpp::BaseBenchmark
{
public:
    HEBERROR_DECLARE_CLASS_NAME(MatMultEIPBenchmark)

public:
    static constexpr std::int64_t tag = 0x20 + MatMultEIPBenchmarkDescription::MatMultEIPID;

    MatMultEIPBenchmark(PalisadeEngine &engine,
                        const hebench::APIBridge::BenchmarkDescriptor &bench_desc,
                        const hebench::APIBridge::WorkloadParams &bench_params);
    ~MatMultEIPBenchmark() override;

    hebench::APIBridge::Handle encode(const hebench::APIBridge::PackedData *p_parameters) override;
    void decode(hebench::APIBridge::Handle encoded_data, hebench::APIBridge::PackedData *p_native) override;
    hebench::APIBridge::Handle encrypt(hebench::APIBridge::Handle encoded_data) override;
    hebench::APIBridge::Handle decrypt(hebench::APIBridge::Handle encrypted_data) override;

    hebench::APIBridge::Handle load(const hebench::APIBridge::Handle *p_local_data, std::uint64_t count) override;
    void store(hebench::APIBridge::Handle remote_data,
               hebench::APIBridge::Handle *p_local_data, std::uint64_t count) override;

    hebench::APIBridge::Handle operate(hebench::APIBridge::Handle h_remote_packed,
                                       const hebench::APIBridge::ParameterIndexer *p_param_indexers) override;

    std::int64_t classTag() const override { return BaseBenchmark::classTag() | MatMultEIPBenchmark::tag; }

private:
    static constexpr std::uint64_t ParametersCount       = 2; // number of parameters for this operation
    static constexpr std::uint64_t ResultComponentsCount = 1; // number of components of result for this operation

    static constexpr std::int64_t tagEncodedResult   = 0x20;
    static constexpr std::int64_t tagEncryptedResult = 0x10;

    template <class T>
    struct InternalMatrix
    {
    public:
        InternalMatrix(std::uint64_t param_position = 0) :
            m_param_position(param_position)
        {
            m_p_rows = std::make_shared<std::vector<T>>();
        }
        const std::vector<T> &rows() const { return *m_p_rows; }
        std::vector<T> &rows() { return *m_p_rows; }
        std::uint64_t paramPosition() const { return m_param_position; }

    private:
        std::shared_ptr<std::vector<T>> m_p_rows;
        std::uint64_t m_param_position;
    };

    typedef InternalMatrix<lbcrypto::Plaintext> InternalMatrixPlain;
    typedef InternalMatrix<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> InternalMatrixCipher;

    static std::vector<std::vector<double>> prepareMatrix(const hebench::APIBridge::NativeDataBuffer &buffer,
                                                          std::uint64_t rows, std::uint64_t cols);
    std::vector<lbcrypto::Plaintext> encodeMatrix(const std::vector<std::vector<double>> &data);
    std::vector<lbcrypto::Plaintext> encodeM0(const std::vector<std::vector<double>> &data);
    std::vector<lbcrypto::Plaintext> encodeM1(const std::vector<std::vector<double>> &data);
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> encryptMatrix(const std::vector<lbcrypto::Plaintext> &plain);
    std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>
    doMatMultEIP(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &M0,
                 const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &M1_T);

    PalisadeContext::Ptr m_p_context;
    hebench::cpp::WorkloadParams::MatrixMultiply m_w_params;
    int m_num_threads;
};

} // namespace ckks
} // namespace pbe
