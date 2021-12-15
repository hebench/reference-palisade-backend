
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>

#include "benchmarks/ckks/palisade_ckks_matmult_cipherbatchaxis_benchmark.h"
#include "engine/palisade_engine.h"
#include "engine/palisade_error.h"
#include <omp.h>

namespace pbe {
namespace ckks {

//-----------------------------------
// class MatMultCipherBatchAxisBenchmarkDescription
//-----------------------------------

MatMultCipherBatchAxisBenchmarkDescription::MatMultCipherBatchAxisBenchmarkDescription()
{
    // initialize the descriptor for this benchmark
    std::memset(&m_descriptor, 0, sizeof(hebench::APIBridge::BenchmarkDescriptor));
    m_descriptor.workload          = hebench::APIBridge::Workload::MatrixMultiply;
    m_descriptor.data_type         = hebench::APIBridge::DataType::Float64;
    m_descriptor.category          = hebench::APIBridge::Category::Latency;
    m_descriptor.cipher_param_mask = HEBENCH_HE_PARAM_FLAGS_ALL_CIPHER;
    //
    m_descriptor.cat_params.latency.min_test_time_ms        = 0;
    m_descriptor.cat_params.latency.warmup_iterations_count = 1;
    //
    m_descriptor.scheme   = HEBENCH_HE_SCHEME_CKKS;
    m_descriptor.security = HEBPALISADE_HE_SECURITY_128;
    m_descriptor.other    = MatMultCipherBatchAxisID;

    // specify default arguments for this workload:
    hebench::cpp::WorkloadParams::MatrixMultiply default_workload_params;
    default_workload_params.rows_M0 = 10;
    default_workload_params.cols_M0 = 9;
    default_workload_params.cols_M1 = 8;
    default_workload_params.add<std::uint64_t>(MatMultCipherBatchAxisBenchmarkDescription::DefaultPolyModulusDegree, "PolyModulusDegree");
    default_workload_params.add<std::uint64_t>(MatMultCipherBatchAxisBenchmarkDescription::DefaultNumCoefficientModuli, "MultiplicativeDepth");
    default_workload_params.add<std::uint64_t>(MatMultCipherBatchAxisBenchmarkDescription::DefaultScaleExponent, "ScaleBits");
    // total: 6 workload params
    this->addDefaultParameters(default_workload_params);
}

MatMultCipherBatchAxisBenchmarkDescription::~MatMultCipherBatchAxisBenchmarkDescription()
{
    //
}

std::string MatMultCipherBatchAxisBenchmarkDescription::getBenchmarkDescription(const hebench::APIBridge::WorkloadParams *p_w_params) const
{
    assert(p_w_params->count >= MatMultCipherBatchAxisBenchmarkDescription::NumWorkloadParams);

    std::size_t pmd        = p_w_params->params[Index_PolyModulusDegree].u_param;
    std::size_t mult_depth = p_w_params->params[Index_NumCoefficientModuli].u_param;
    std::size_t scale_bits = p_w_params->params[Index_ScaleExponent].u_param;

    std::stringstream ss;
    ss << ", Encryption parameters" << std::endl
       << ", , HE Library, PALISADE 1.11.3" << std::endl
       << ", , Key-switching technique, PALISADE Hybrid" << std::endl
       << ", , Poly modulus degree, " << pmd << std::endl
       << ", , Multiplicative Depth, " << mult_depth << std::endl
       << ", , Scale, 2^" << scale_bits << std::endl
       << ", Algorithm, " << AlgorithmName << ", " << AlgorithmDescription;
    return ss.str();
}

hebench::cpp::BaseBenchmark *MatMultCipherBatchAxisBenchmarkDescription::createBenchmark(hebench::cpp::BaseEngine &engine,
                                                                                         const hebench::APIBridge::WorkloadParams *p_params)
{
    if (!p_params)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid empty workload parameters. This workload requires flexible parameters."),
                                         HEBENCH_ECODE_CRITICAL_ERROR);

    PalisadeEngine &ex_engine = dynamic_cast<PalisadeEngine &>(engine);
    return new MatMultCipherBatchAxisBenchmark(ex_engine, m_descriptor, *p_params);
}

void MatMultCipherBatchAxisBenchmarkDescription::destroyBenchmark(hebench::cpp::BaseBenchmark *p_bench)
{
    MatMultCipherBatchAxisBenchmark *p = dynamic_cast<MatMultCipherBatchAxisBenchmark *>(p_bench);
    if (p)
        delete p;
}

//------------------------
// class MatMultCipherBatchAxisBenchmark
//------------------------

MatMultCipherBatchAxisBenchmark::MatMultCipherBatchAxisBenchmark(PalisadeEngine &engine,
                                                                 const hebench::APIBridge::BenchmarkDescriptor &bench_desc,
                                                                 const hebench::APIBridge::WorkloadParams &bench_params) :
    hebench::cpp::BaseBenchmark(engine, bench_desc, bench_params),
    m_w_params(bench_params)
{
    // validate workload parameters

    // number of workload parameters (3 for matmult; +3 encryption params)
    if (bench_params.count < MatMultCipherBatchAxisBenchmarkDescription::NumWorkloadParams)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid workload parameters. This workload requires " + std::to_string(MatMultCipherBatchAxisBenchmarkDescription::NumWorkloadParams) + " parameters."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    std::size_t pmd        = m_w_params.get<std::uint64_t>(MatMultCipherBatchAxisBenchmarkDescription::Index_PolyModulusDegree);
    std::size_t mult_depth = m_w_params.get<std::uint64_t>(MatMultCipherBatchAxisBenchmarkDescription::Index_NumCoefficientModuli);
    std::size_t scale_bits = m_w_params.get<std::uint64_t>(MatMultCipherBatchAxisBenchmarkDescription::Index_ScaleExponent);

    // check values of the workload parameters and make sure they are supported by benchmark:

    if (m_w_params.rows_M0 <= 0 || m_w_params.cols_M0 <= 0 || m_w_params.cols_M1 <= 0)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Matrix dimensions must be greater than 0."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    if (m_w_params.cols_M0 - 1 > pmd / 2)
    {
        std::stringstream ss;
        ss << "Invalid workload parameters. This workload only supports matrices of dimensions (n x "
           << (pmd / 2) << ") x (" << (pmd / 2) << " x m).";
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                         HEBENCH_ECODE_INVALID_ARGS);
    } // end if

    //    std::vector<int32_t> vec(m_w_params.cols_M0);
    //    std::iota(std::begin(vec), std::end(vec), 1);
    m_p_context = PalisadeContext::createCKKSContext(pmd, mult_depth, scale_bits, lbcrypto::HEStd_128_classic, m_w_params.cols_M0);
    m_p_context->EvalMultKeyGen();
    //    m_p_context->EvalAtIndexKeyGen(vec);
    //    m_p_context->EvalSumKeyGen();
}

MatMultCipherBatchAxisBenchmark::~MatMultCipherBatchAxisBenchmark()
{
    // nothing needed in this example
}

//--------------------------
// Provided methods - Start
//--------------------------

//--------------------------
// Provided methods - End
//--------------------------

hebench::APIBridge::Handle MatMultCipherBatchAxisBenchmark::encode(const hebench::APIBridge::PackedData *p_parameters)
{
    // since this benchmark is cipher-cipher, encode receives 2 parameter packs from test harness

    if (p_parameters->pack_count != MatMultCipherBatchAxisBenchmarkDescription::OpParametersCount)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Expected 2 parameter packs, but " + std::to_string(p_parameters->pack_count) + " received."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    std::vector<OpParamSamplePlain> retval;
    // - for latency operation, we have a single sample per data pack
    retval.emplace_back(m_w_params.rows_M0, m_w_params.cols_M0); // op param 0
    retval.emplace_back(m_w_params.cols_M0, m_w_params.cols_M1); // op param 1

    for (std::uint64_t op_param_i = 0; op_param_i < MatMultCipherBatchAxisBenchmarkDescription::OpParametersCount; ++op_param_i)
    {
        // find data pack corresponding to this op parameter
        const hebench::APIBridge::DataPack &data_pack = MatMultCipherBatchAxisBenchmark::findDataPack(*p_parameters, op_param_i);
        if (data_pack.buffer_count < 1)
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Latency test requires, at least, 1 sample per operation parameter. None found for operation parameter " + std::to_string(op_param_i) + "."),
                                             HEBENCH_ECODE_INVALID_ARGS);
        if (!data_pack.p_buffers)
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Unexpected empty buffer in data pack."),
                                             HEBENCH_ECODE_CRITICAL_ERROR);

        const double *p_raw_buffer = reinterpret_cast<double *>(data_pack.p_buffers[0].p);
        if (!p_raw_buffer
            || data_pack.p_buffers[0].size < retval[op_param_i].rows() * retval[op_param_i].cols() * sizeof(double))
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Insufficient data in buffer for operation parameter " + std::to_string(op_param_i) + "."),
                                             HEBENCH_ECODE_CRITICAL_ERROR);
        for (std::uint64_t col_i = 0; col_i < retval[op_param_i].cols(); ++col_i)
        {
            for (std::uint64_t row_i = 0; row_i < retval[op_param_i].rows(); ++row_i)
            {
                // encode this op param in column major format (incoming raw is row major)
                double clear_value                 = p_raw_buffer[row_i * retval[op_param_i].cols() + col_i];
                lbcrypto::Plaintext &plain_encoded = retval[op_param_i].at(row_i, col_i);
                plain_encoded                      = m_p_context->context()->MakeCKKSPackedPlaintext(std::vector<double>(1, clear_value));
            } // end for
        } // end for
    } // end for

    // return encoded data as handle
    return this->getEngine().template createHandle<decltype(retval)>(sizeof(retval),
                                                                     0,
                                                                     std::move(retval));
}

void MatMultCipherBatchAxisBenchmark::decode(hebench::APIBridge::Handle h_encoded_data, hebench::APIBridge::PackedData *p_native)
{
    // able to decode only encoded result

    assert(p_native && p_native->p_data_packs && p_native->pack_count > 0);

    if (p_native->pack_count > 0)
    {
        if (!p_native->p_data_packs)
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Unexpected empty 'p_native->p_data_packs'."),
                                             HEBENCH_ECODE_CRITICAL_ERROR);

        std::vector<OpParamSamplePlain> &encoded =
            this->getEngine().retrieveFromHandle<std::vector<OpParamSamplePlain>>(h_encoded_data);

        for (std::size_t op_param_i = 0; op_param_i < encoded.size(); ++op_param_i)
        {
            // find data pack corresponding to this op parameter
            std::uint64_t pack_idx                    = MatMultCipherBatchAxisBenchmark::findDataPackIndex(*p_native, op_param_i);
            hebench::APIBridge::DataPack *p_data_pack = pack_idx < p_native->pack_count ?
                                                            &p_native->p_data_packs[pack_idx] :
                                                            nullptr;
            if (p_data_pack && p_data_pack->buffer_count > 0)
            {
                if (!p_data_pack->p_buffers)
                    throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Unexpected empty buffer in data pack."),
                                                     HEBENCH_ECODE_CRITICAL_ERROR);

                if (p_data_pack->p_buffers[0].size > 0)
                {
                    if (!p_data_pack->p_buffers[0].p)
                        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Unexpected empty buffer in data pack."),
                                                         HEBENCH_ECODE_CRITICAL_ERROR);
                    double *result_clear           = reinterpret_cast<double *>(p_data_pack->p_buffers[0].p);
                    double *result_clear_begin     = result_clear;
                    const double *result_clear_end = result_clear_begin + p_data_pack->p_buffers[0].size / sizeof(double);
                    // decode as much as we can (coverting from column major to row major)
                    auto it = result_clear_begin;
                    for (std::uint64_t row_i = 0; it != result_clear_end && row_i < encoded[op_param_i].rows(); ++row_i)
                    {
                        for (std::uint64_t col_i = 0; it != result_clear_end && col_i < encoded[op_param_i].cols(); ++col_i)
                        {
                            std::vector<double> clear_decoded =
                                encoded[op_param_i].at(row_i, col_i)->GetRealPackedValue();
                            if (clear_decoded.empty() || std::abs(clear_decoded.front()) < 0.00005)
                                *it = 0.0;
                            else
                                *it = clear_decoded.front();
                            ++it;
                        } // end for
                    } // end for
                } // end if
            } // end if
        } // end if
    } // end if
}

hebench::APIBridge::Handle MatMultCipherBatchAxisBenchmark::encrypt(hebench::APIBridge::Handle h_encoded_data)
{
    std::vector<OpParamSamplePlain> &encoded =
        this->getEngine().retrieveFromHandle<std::vector<OpParamSamplePlain>>(h_encoded_data);

    std::vector<OpParamSampleCipher> retval;
    for (auto plain_it = encoded.begin(); plain_it != encoded.end(); ++plain_it)
    {
        retval.emplace_back(plain_it->rows(), plain_it->cols());
        for (std::uint64_t col_i = 0; col_i < plain_it->cols(); ++col_i)
        {
            for (std::uint64_t row_i = 0; row_i < plain_it->rows(); ++row_i)
            {

                retval.back().at(row_i, col_i) =
                    m_p_context->context()->Encrypt(m_p_context->publicKey(), plain_it->at(row_i, col_i));
            } // end for
        } // end for
    } // end for

    return this->getEngine().createHandle<decltype(retval)>(sizeof(retval),
                                                            0,
                                                            std::move(retval));
}

hebench::APIBridge::Handle MatMultCipherBatchAxisBenchmark::decrypt(hebench::APIBridge::Handle h_encrypted_data)
{
    // supports only encrypted result
    std::vector<OpParamSampleCipher> &encrypted =
        this->getEngine().retrieveFromHandle<std::vector<OpParamSampleCipher>>(h_encrypted_data);

    std::vector<OpParamSamplePlain> retval;
    for (auto cipher_it = encrypted.begin(); cipher_it != encrypted.end(); ++cipher_it)
    {
        retval.emplace_back(cipher_it->rows(), cipher_it->cols());
        for (std::uint64_t col_i = 0; col_i < cipher_it->cols(); ++col_i)
        {
            for (std::uint64_t row_i = 0; row_i < cipher_it->rows(); ++row_i)
            {
                m_p_context->decrypt(cipher_it->at(row_i, col_i), retval.back().at(row_i, col_i));
            } // end for
        } // end for
    } // end for

    return this->getEngine().createHandle<decltype(retval)>(sizeof(retval),
                                                            0,
                                                            std::move(retval));
}

hebench::APIBridge::Handle MatMultCipherBatchAxisBenchmark::load(const hebench::APIBridge::Handle *p_local_data, uint64_t count)
{
    if (count != 1)
        // we do ops in ciphertext only, so we should get 1 pack
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid number of handles. Expected 1."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    // host is same as remote, so, just duplicate handle to let called be able to destroy input handle
    return this->getEngine().duplicateHandle(p_local_data[0]);
}

void MatMultCipherBatchAxisBenchmark::store(hebench::APIBridge::Handle h_remote_data,
                                            hebench::APIBridge::Handle *p_local_data, std::uint64_t count)
{
    // supports only storing result
    if (count > 0 && !p_local_data)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid null array of handles: \"p_local_data\""),
                                         HEBENCH_ECODE_INVALID_ARGS);

    std::memset(p_local_data, 0, sizeof(hebench::APIBridge::Handle) * count);

    if (count > 0)
    {
        // host is same as remote, so, just duplicate handle to let called be able to destroy input handle
        p_local_data[0] = this->getEngine().duplicateHandle(h_remote_data);
    } // end if
}

hebench::APIBridge::Handle MatMultCipherBatchAxisBenchmark::operate(hebench::APIBridge::Handle h_remote_packed,
                                                                    const hebench::APIBridge::ParameterIndexer *p_param_indexers)
{
    for (std::size_t i = 0; i < MatMultCipherBatchAxisBenchmarkDescription::OpParametersCount; ++i)
    {
        if (p_param_indexers[i].value_index > 0)
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Unexpected index in parameter indexer."),
                                             HEBENCH_ECODE_INVALID_ARGS);
        if (p_param_indexers[i].batch_size > 1)
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Batch size must be 1 for latency test."),
                                             HEBENCH_ECODE_INVALID_ARGS);
    } // end for

    std::vector<OpParamSampleCipher> &remote =
        this->getEngine().retrieveFromHandle<std::vector<OpParamSampleCipher>>(h_remote_packed);

    if (remote.size() < 2)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Insufficient number of arguments for operation parameters."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    const OpParamSampleCipher &m0 = remote[0];
    const OpParamSampleCipher &m1 = remote[1];
    std::vector<OpParamSampleCipher> retval;
    retval.emplace_back(m0.rows(), m1.cols());
    OpParamSampleCipher &m = retval.back();
    int th_lvl             = m0.rows() * m1.cols();
    if (th_lvl <= 0)
        th_lvl = 1;
    else if (th_lvl > omp_get_max_threads())
        th_lvl = omp_get_max_threads();

    std::exception_ptr p_ex;
    std::mutex mtx_ex;

#pragma omp parallel for num_threads(th_lvl) collapse(2)
    for (size_t out_ind0 = 0; out_ind0 < m0.rows(); ++out_ind0)
    {
        for (size_t out_ind1 = 0; out_ind1 < m1.cols(); ++out_ind1)
        {
            try
            {
                if (!p_ex)
                {
                    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &out = m.at(out_ind0, out_ind1);
                    for (size_t inner_dim = 0; inner_dim < m0.cols(); inner_dim++)
                    {
                        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &arg1 = m0.at(out_ind0, inner_dim);
                        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &arg2 = m1.at(inner_dim, out_ind1);

                        if (inner_dim == 0)
                            out = m_p_context->context()->EvalMultNoRelin(arg1, arg2);
                        else
                            out += m_p_context->context()->EvalMultNoRelin(arg1, arg2);
                    }
                } // end if
            }
            catch (...)
            {
                std::scoped_lock<std::mutex> lock(mtx_ex);
                if (!p_ex)
                    p_ex = std::current_exception();
            }
        }
    }
    if (p_ex)
        std::rethrow_exception(p_ex);

    p_ex = nullptr;

#pragma omp parallel for
    for (size_t out_idx = 0; out_idx < m.size(); ++out_idx)
    {
        try
        {
            if (!p_ex)
            {
                m.at(out_idx) = m_p_context->context()->Relinearize(m.at(out_idx));
                m.at(out_idx) = m_p_context->context()->Rescale(m.at(out_idx));
            } // end if
        }
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(mtx_ex);
            if (!p_ex)
                p_ex = std::current_exception();
        }
    }
    if (p_ex)
        std::rethrow_exception(p_ex);

    return this->getEngine().createHandle<decltype(retval)>(sizeof(retval),
                                                            0,
                                                            std::move(retval));
}

} // namespace ckks
} // namespace pbe
