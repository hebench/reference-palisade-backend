
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "benchmarks/bfv/palisade_bfv_dot_product_benchmark.h"
#include "engine/palisade_engine.h"
#include "engine/palisade_error.h"
#include <omp.h>

using namespace pbe::bfv;

//-----------------------------------
// class DotProductBenchmarkDescription
//-----------------------------------

DotProductBenchmarkDescription::DotProductBenchmarkDescription(hebench::APIBridge::Category category)
{
    // initialize the descriptor for this benchmark
    std::memset(&m_descriptor, 0, sizeof(hebench::APIBridge::BenchmarkDescriptor));
    m_descriptor.data_type = hebench::APIBridge::DataType::Int64;
    m_descriptor.category  = category;
    switch (category)
    {
    case hebench::APIBridge::Category::Latency:
        m_descriptor.cat_params.latency.min_test_time_ms        = 0;
        m_descriptor.cat_params.latency.warmup_iterations_count = 1;
        break;

    case hebench::APIBridge::Category::Offline:
        m_descriptor.cat_params.offline.data_count[0] = 0;
        m_descriptor.cat_params.offline.data_count[1] = 0;
        break;

    default:
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid category received."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    }
    m_descriptor.cipher_param_mask = HEBENCH_HE_PARAM_FLAGS_ALL_CIPHER;
    //
    m_descriptor.scheme   = HEBENCH_HE_SCHEME_BFV;
    m_descriptor.security = HEBPALISADE_HE_SECURITY_128;
    m_descriptor.other    = 0; // no extra parameters
    m_descriptor.workload = hebench::APIBridge::Workload::DotProduct;

    hebench::cpp::WorkloadParams::DotProduct default_workload_params;
    default_workload_params.n = 100;
    default_workload_params.add<std::uint64_t>(DotProductBenchmarkDescription::DefaultPolyModulusDegree, "PolyModulusDegree");
    default_workload_params.add<std::uint64_t>(DotProductBenchmarkDescription::DefaultMultiplicativeDepth, "MultiplicativeDepth");
    default_workload_params.add<std::uint64_t>(DotProductBenchmarkDescription::DefaultCoeffModulusBits, "CoefficientModulusBits");
    this->addDefaultParameters(default_workload_params);
}

DotProductBenchmarkDescription::~DotProductBenchmarkDescription()
{
    // nothing needed in this example
}

hebench::cpp::BaseBenchmark *DotProductBenchmarkDescription::createBenchmark(hebench::cpp::BaseEngine &engine, const hebench::APIBridge::WorkloadParams *p_params)
{
    return new DotProductBenchmark(engine, m_descriptor, *p_params);
}

void DotProductBenchmarkDescription::destroyBenchmark(hebench::cpp::BaseBenchmark *p_bench)
{
    if (p_bench)
        delete p_bench;
}

std::string DotProductBenchmarkDescription::getBenchmarkDescription(const hebench::APIBridge::WorkloadParams *p_w_params) const
{
    std::stringstream ss;
    std::string s_tmp = BenchmarkDescription::getBenchmarkDescription(p_w_params);

    if (!p_w_params)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid null workload parameters `p_w_params`"),
                                         HEBENCH_ECODE_INVALID_ARGS);

    std::size_t pmd        = p_w_params->params[DotProductBenchmarkDescription::Index_PolyModulusDegree].u_param;
    std::size_t mult_depth = p_w_params->params[DotProductBenchmarkDescription::Index_NumCoefficientModuli].u_param;
    std::size_t coeff_bits = p_w_params->params[DotProductBenchmarkDescription::Index_CoefficientModulusBits].u_param;
    if (!s_tmp.empty())
        ss << s_tmp << std::endl;
    ss << ", Encryption parameters" << std::endl
       << ", , HE Library, PALISADE 1.11.3" << std::endl
       << ", , Poly modulus degree, " << pmd << std::endl
       << ", , Multiplicative Depth, " << mult_depth << std::endl
       << ", , Coefficient moduli bits, " << coeff_bits << std::endl
       << ", Algorithm, " << AlgorithmName << ", " << AlgorithmDescription;

    return ss.str();
}

//------------------------
// DotProductBenchmark
//------------------------

DotProductBenchmark::DotProductBenchmark(hebench::cpp::BaseEngine &engine,
                                         const hebench::APIBridge::BenchmarkDescriptor &bench_desc,
                                         const hebench::APIBridge::WorkloadParams &bench_params) :
    hebench::cpp::BaseBenchmark(engine, bench_desc, bench_params),
    m_w_params(bench_params)
{
    assert(bench_params.count >= DotProductBenchmarkDescription::NumWorkloadParams);

    if (m_w_params.n <= 0)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Vector size must be greater than 0."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    std::size_t poly_modulus_degree  = m_w_params.get<std::uint64_t>(DotProductBenchmarkDescription::Index_PolyModulusDegree);
    std::size_t multiplicative_depth = m_w_params.get<std::uint64_t>(DotProductBenchmarkDescription::Index_NumCoefficientModuli);
    std::size_t coeff_mudulus_bits   = m_w_params.get<std::uint64_t>(DotProductBenchmarkDescription::Index_CoefficientModulusBits);

    if (coeff_mudulus_bits < 1)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Multiplicative depth must be greater than 0."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    m_p_ctx_wrapper = PalisadeContext::createBFVContext(poly_modulus_degree,
                                                        multiplicative_depth,
                                                        coeff_mudulus_bits,
                                                        lbcrypto::HEStd_128_classic);
    m_p_ctx_wrapper->EvalMultKeyGen();
    m_p_ctx_wrapper->EvalSumKeyGen();
    std::size_t slot_count = m_p_ctx_wrapper->getSlotCount();
    if (m_w_params.n > slot_count)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Vector size cannot be greater than " + std::to_string(slot_count) + "."),
                                         HEBENCH_ECODE_INVALID_ARGS);
}

DotProductBenchmark::~DotProductBenchmark()
{
    // nothing needed in this example
}

hebench::APIBridge::Handle DotProductBenchmark::encode(const hebench::APIBridge::PackedData *p_parameters)
{
    if (p_parameters->pack_count != DotProductBenchmarkDescription::NumOpParams)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid number of parameters detected in parameter pack. Expected 2."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    std::vector<std::vector<lbcrypto::Plaintext>> params;

    params.resize(p_parameters->pack_count);
    const unsigned int params_size = params.size();
    for (unsigned int x = 0; x < params_size; ++x)
    {
        params[x].resize(p_parameters->p_data_packs[x].buffer_count);
    }

    std::vector<int64_t> values;
    values.resize(m_w_params.n);
    for (unsigned int x = 0; x < params.size(); ++x)
    {
        for (unsigned int y = 0; y < params[x].size(); ++y)
        {
            const hebench::APIBridge::DataPack &parameter = p_parameters->p_data_packs[x];
            // take first sample from parameter (because latency test has a single sample per parameter)
            const hebench::APIBridge::NativeDataBuffer &sample = parameter.p_buffers[y];
            // convert the native data to pointer to int64_t as per specification of workload
            const int64_t *p_row = reinterpret_cast<const int64_t *>(sample.p);
            for (unsigned int x = 0; x < m_w_params.n; ++x)
            {
                values[x] = p_row[x];
            }
            params[x][y] = m_p_ctx_wrapper->context()->MakePackedPlaintext(values);
        }
    }

    return this->getEngine().createHandle<decltype(params)>(sizeof(params),
                                                            0,
                                                            std::move(params));
}

void DotProductBenchmark::decode(hebench::APIBridge::Handle encoded_data, hebench::APIBridge::PackedData *p_native)
{
    // retrieve our internal format object from the handle
    const std::vector<lbcrypto::Plaintext> &params =
        this->getEngine().retrieveFromHandle<std::vector<lbcrypto::Plaintext>>(encoded_data);

    const size_t params_size = params.size();
    for (size_t result_i = 0; result_i < params_size; ++result_i)
    {
        int64_t *output_location = reinterpret_cast<int64_t *>(p_native->p_data_packs[0].p_buffers[result_i].p);
        std::vector<int64_t> result_vec;
        //m_p_ctx_wrapper->BFVEncoder()->decode(params[result_i], result_vec); TODO REMOVE
        result_vec         = params[result_i]->GetPackedValue();
        output_location[0] = result_vec.front();
    }
}

hebench::APIBridge::Handle DotProductBenchmark::encrypt(hebench::APIBridge::Handle encoded_data)
{
    // retrieve our internal format object from the handle
    const std::vector<std::vector<lbcrypto::Plaintext>> &encoded_data_ref =
        this->getEngine().retrieveFromHandle<std::vector<std::vector<lbcrypto::Plaintext>>>(encoded_data);

    std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> encrypted_data;
    encrypted_data.resize(encoded_data_ref.size());
    for (unsigned int param_i = 0; param_i < encoded_data_ref.size(); param_i++)
    {
        encrypted_data[param_i].resize(encoded_data_ref[param_i].size());
        for (unsigned int parameter_sample = 0; parameter_sample < encoded_data_ref[param_i].size(); parameter_sample++)
        {
            //m_p_ctx_wrapper->encryptor()->encrypt(encoded_data_ref[param_i][parameter_sample], encrypted_data[param_i][parameter_sample]);
            encrypted_data[param_i][parameter_sample] = m_p_ctx_wrapper->context()->Encrypt(m_p_ctx_wrapper->publicKey(), encoded_data_ref[param_i][parameter_sample]);
        }
    }

    return this->getEngine().createHandle<decltype(encrypted_data)>(sizeof(encrypted_data),
                                                                    0,
                                                                    std::move(encrypted_data));
}

hebench::APIBridge::Handle DotProductBenchmark::decrypt(hebench::APIBridge::Handle encrypted_data)
{
    // retrieve our internal format object from the handle
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &encrypted_data_ref =
        this->getEngine().retrieveFromHandle<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>(encrypted_data);

    std::vector<lbcrypto::Plaintext> plaintext_data;
    plaintext_data.resize(encrypted_data_ref.size());

    for (unsigned int res_count = 0; res_count < encrypted_data_ref.size(); ++res_count)
    {
        plaintext_data[res_count] = m_p_ctx_wrapper->decrypt(encrypted_data_ref[res_count]);
    }

    return this->getEngine().createHandle<decltype(plaintext_data)>(sizeof(plaintext_data),
                                                                    0,
                                                                    std::move(plaintext_data));
}

hebench::APIBridge::Handle DotProductBenchmark::load(const hebench::APIBridge::Handle *p_local_data, uint64_t count)
{
    assert(count == 0 || p_local_data);
    if (count != 1)
        // we do all ops in plain text, so, we should get only one pack of data
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid number of handles. Expected 1."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    return this->getEngine().duplicateHandle(p_local_data[0]);
}

void DotProductBenchmark::store(hebench::APIBridge::Handle remote_data,
                                hebench::APIBridge::Handle *p_local_data, std::uint64_t count)
{
    assert(count == 0 || p_local_data);
    if (count > 0)
    {
        // pad with zeros any excess local handles as per specifications
        std::memset(p_local_data, 0, sizeof(hebench::APIBridge::Handle) * count);

        // since remote and host are the same, we just need to return a copy
        // of the remote as local data.
        p_local_data[0] = this->getEngine().duplicateHandle(remote_data);
    } // end if
}

hebench::APIBridge::Handle DotProductBenchmark::operate(hebench::APIBridge::Handle h_remote_packed,
                                                        const hebench::APIBridge::ParameterIndexer *p_param_indexers)
{
    // retrieve our internal format object from the handle
    const std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> &params =
        this->getEngine().retrieveFromHandle<std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>>(h_remote_packed);

    // create a new internal object for result
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> result;
    // perform the actual operation
    result.resize(p_param_indexers[0].batch_size * p_param_indexers[1].batch_size);
    std::mutex mtx;
    std::exception_ptr p_ex;
#pragma omp parallel for collapse(2)
    for (uint64_t result_i = 0; result_i < p_param_indexers[0].batch_size; result_i++)
    {
        for (uint64_t result_x = 0; result_x < p_param_indexers[1].batch_size; result_x++)
        {
            try
            {
                if (!p_ex)
                {
                    auto &result_cipher = result[result_i * p_param_indexers[1].batch_size + result_x];
                    result_cipher       = m_p_ctx_wrapper->context()->EvalMultAndRelinearize(params[0][p_param_indexers[0].value_index + result_i],
                                                                                       params[1][p_param_indexers[1].value_index + result_x]);

                    result_cipher = m_p_ctx_wrapper->context()->EvalSum(result_cipher, m_w_params.n);
                } // end if
            }
            catch (...)
            {
                std::scoped_lock<std::mutex> lock(mtx);
                if (!p_ex)
                    p_ex = std::current_exception();
            }
        }
    }
    if (p_ex)
        std::rethrow_exception(p_ex);

    return this->getEngine().createHandle<decltype(result)>(sizeof(result),
                                                            0,
                                                            std::move(result));
}
