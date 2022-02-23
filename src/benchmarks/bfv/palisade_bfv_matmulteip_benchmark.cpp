
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>

#include <omp.h>

#include "benchmarks/bfv/palisade_bfv_matmulteip_benchmark.h"
#include "engine/palisade_engine.h"
#include "engine/palisade_error.h"

namespace pbe {
namespace bfv {

//-----------------------------------
// class MatMultEIPBenchmarkDescription
//-----------------------------------

MatMultEIPBenchmarkDescription::MatMultEIPBenchmarkDescription()
{
    // initialize the descriptor for this benchmark
    std::memset(&m_descriptor, 0, sizeof(hebench::APIBridge::BenchmarkDescriptor));
    m_descriptor.workload          = hebench::APIBridge::Workload::MatrixMultiply;
    m_descriptor.data_type         = hebench::APIBridge::DataType::Int64;
    m_descriptor.category          = hebench::APIBridge::Category::Latency;
    m_descriptor.cipher_param_mask = HEBENCH_HE_PARAM_FLAGS_ALL_CIPHER;
    //
    m_descriptor.cat_params.latency.min_test_time_ms        = 0;
    m_descriptor.cat_params.latency.warmup_iterations_count = 1;
    //
    m_descriptor.scheme   = HEBENCH_HE_SCHEME_BFV;
    m_descriptor.security = HEBPALISADE_HE_SECURITY_128;
    m_descriptor.other    = MatMultEIPID;

    // specify default arguments for this workload:
    hebench::cpp::WorkloadParams::MatrixMultiply default_workload_params;
    default_workload_params.rows_M0() = 10;
    default_workload_params.cols_M0() = 9;
    default_workload_params.cols_M1() = 8;
    default_workload_params.add<std::uint64_t>(MatMultEIPBenchmarkDescription::DefaultPolyModulusDegree, "PolyModulusDegree");
    default_workload_params.add<std::uint64_t>(MatMultEIPBenchmarkDescription::DefaultNumCoefficientModuli, "MultiplicativeDepth");
    default_workload_params.add<std::uint64_t>(MatMultEIPBenchmarkDescription::DefaultCoefficientModuliBits, "CoefficientModuliBits");
    default_workload_params.add<std::uint64_t>(MatMultEIPBenchmarkDescription::DefaultNumThreads, "NumThreads");
    // total: 7 workload params
    this->addDefaultParameters(default_workload_params);
}

MatMultEIPBenchmarkDescription::~MatMultEIPBenchmarkDescription()
{
    //
}

std::string MatMultEIPBenchmarkDescription::getBenchmarkDescription(const hebench::APIBridge::WorkloadParams *p_w_params) const
{
    std::stringstream ss;
    std::string s_tmp = BenchmarkDescription::getBenchmarkDescription(p_w_params);

    if (!p_w_params)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid null workload parameters `p_w_params`"),
                                         HEBENCH_ECODE_INVALID_ARGS);

    assert(p_w_params->count >= MatMultEIPBenchmarkDescription::NumWorkloadParams);

    std::size_t pmd           = p_w_params->params[Index_PolyModulusDegree].u_param;
    std::size_t mult_depth    = p_w_params->params[Index_NumCoefficientModuli].u_param;
    std::size_t coeff_bits    = p_w_params->params[Index_CoefficientModuliBits].u_param;
    std::uint64_t num_threads = p_w_params->params[MatMultEIPBenchmarkDescription::Index_NumThreads].u_param;

    if (num_threads <= 0)
        num_threads = omp_get_max_threads();
    if (!s_tmp.empty())
        ss << s_tmp << std::endl;
    ss << ", Encryption parameters" << std::endl
       << ", , HE Library, PALISADE 1.11.3" << std::endl
       << ", , Poly modulus degree, " << pmd << std::endl
       << ", , Multiplicative Depth, " << mult_depth << std::endl
       << ", , Coefficient moduli bits, " << coeff_bits << std::endl
       << ", Algorithm, " << AlgorithmName << ", " << AlgorithmDescription << std::endl
       << ", Number of threads, " << num_threads;
    return ss.str();
}

hebench::cpp::BaseBenchmark *MatMultEIPBenchmarkDescription::createBenchmark(hebench::cpp::BaseEngine &engine,
                                                                             const hebench::APIBridge::WorkloadParams *p_params)
{
    if (!p_params)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid empty workload parameters. This workload requires flexible parameters."),
                                         HEBENCH_ECODE_CRITICAL_ERROR);

    PalisadeEngine &ex_engine = dynamic_cast<PalisadeEngine &>(engine);
    return new MatMultEIPBenchmark(ex_engine, m_descriptor, *p_params);
}

void MatMultEIPBenchmarkDescription::destroyBenchmark(hebench::cpp::BaseBenchmark *p_bench)
{
    MatMultEIPBenchmark *p = dynamic_cast<MatMultEIPBenchmark *>(p_bench);
    if (p)
        delete p;
}

//------------------------
// class MatMultEIPBenchmark
//------------------------

MatMultEIPBenchmark::MatMultEIPBenchmark(PalisadeEngine &engine,
                                         const hebench::APIBridge::BenchmarkDescriptor &bench_desc,
                                         const hebench::APIBridge::WorkloadParams &bench_params) :
    hebench::cpp::BaseBenchmark(engine, bench_desc, bench_params),
    m_w_params(bench_params)
{
    // validate workload parameters

    // number of workload parameters (3 for matmult; +3 encryption params)
    if (bench_params.count < MatMultEIPBenchmarkDescription::NumWorkloadParams)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid workload parameters. This workload requires " + std::to_string(MatMultEIPBenchmarkDescription::NumWorkloadParams) + " parameters."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    std::size_t pmd        = m_w_params.get<std::uint64_t>(MatMultEIPBenchmarkDescription::Index_PolyModulusDegree);
    std::size_t mult_depth = m_w_params.get<std::uint64_t>(MatMultEIPBenchmarkDescription::Index_NumCoefficientModuli);
    std::size_t coeff_bits = m_w_params.get<std::uint64_t>(MatMultEIPBenchmarkDescription::Index_CoefficientModuliBits);
    m_num_threads          = static_cast<int>(m_w_params.get<std::uint64_t>(MatMultEIPBenchmarkDescription::Index_NumThreads));
    if (m_num_threads <= 0)
        m_num_threads = omp_get_max_threads();

    // check values of the workload parameters and make sure they are supported by benchmark:

    if (m_w_params.rows_M0() <= 0 || m_w_params.cols_M0() <= 0 || m_w_params.cols_M1() <= 0)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Matrix dimensions must be greater than 0."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    if (m_w_params.cols_M0() > pmd - 1)
    {
        std::stringstream ss;
        ss << "Invalid workload parameters. This workload only supports matrices of dimensions (n x "
           << (pmd - 1) << ") x (" << (pmd - 1) << " x m).";
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                         HEBENCH_ECODE_INVALID_ARGS);
    } // end if

    std::vector<int32_t> vec(m_w_params.cols_M0());
    std::iota(std::begin(vec), std::end(vec), 1);
    m_p_context = PalisadeContext::createBFVContext(pmd, mult_depth, coeff_bits, lbcrypto::HEStd_128_classic);
    m_p_context->EvalMultKeyGen();
    m_p_context->EvalAtIndexKeyGen(vec);
    m_p_context->EvalSumKeyGen();
}

MatMultEIPBenchmark::~MatMultEIPBenchmark()
{
    // nothing needed in this example
}

//--------------------------
// Provided methods - Start
//--------------------------

std::vector<std::vector<std::int64_t>> MatMultEIPBenchmark::prepareMatrix(const hebench::APIBridge::NativeDataBuffer &buffer,
                                                                          std::uint64_t rows, std::uint64_t cols)
{
    std::vector<std::vector<std::int64_t>> retval(rows, std::vector<std::int64_t>(cols));
    if (!buffer.p || buffer.size < rows * cols * sizeof(std::int64_t))
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Insufficient data for M0."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    const std::int64_t *p_curr_row = reinterpret_cast<const std::int64_t *>(buffer.p);
    for (std::size_t row_i = 0; row_i < rows; ++row_i)
    {
        std::copy(p_curr_row, p_curr_row + cols, retval[row_i].begin());
        p_curr_row += cols;
    } // end for
    return retval;
}

std::vector<lbcrypto::Plaintext> MatMultEIPBenchmark::encodeMatrix(const std::vector<std::vector<std::int64_t>> &data)
{
    std::vector<lbcrypto::Plaintext> retval(data.size());
    std::size_t num_cols = data.empty() ? 0 : data.front().size();
    for (size_t i = 0; i < data.size(); ++i)
    {
        if (data[i].size() != num_cols)
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Non-uniform number of columns found in matrix."),
                                             HEBENCH_ECODE_INVALID_ARGS);
        retval[i] = m_p_context->context()->MakePackedPlaintext(data[i]);
    } // end for
    return retval;
}

std::vector<lbcrypto::Plaintext> MatMultEIPBenchmark::encodeM0(const std::vector<std::vector<std::int64_t>> &data)
{
    assert(data.size() == m_w_params.rows_M0());
    assert(!data.empty() && data.front().size() == m_w_params.cols_M0());
    return encodeMatrix(data);
}

std::vector<lbcrypto::Plaintext> MatMultEIPBenchmark::encodeM1(const std::vector<std::vector<std::int64_t>> &data)
{
    assert(data.size() == m_w_params.cols_M0());

    // transpose
    std::vector<std::vector<std::int64_t>> data_T(m_w_params.cols_M1(), std::vector<std::int64_t>(m_w_params.cols_M0()));
    for (size_t row_i = 0; row_i < data.size(); ++row_i)
    {
        assert(data[row_i].size() == m_w_params.cols_M1());
        for (size_t col_i = 0; col_i < data[row_i].size(); ++col_i)
            data_T[col_i][row_i] = data[row_i][col_i];
    } // end if
    return encodeMatrix(data_T);
}

std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> MatMultEIPBenchmark::encryptMatrix(const std::vector<lbcrypto::Plaintext> &plain)
{
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> retval(plain.size());
    for (size_t i = 0; i < plain.size(); ++i)
    {
        retval[i] = m_p_context->context()->Encrypt(m_p_context->publicKey(), plain[i]);
    } // end for
    return retval;
}

std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>
MatMultEIPBenchmark::doMatMultEIP(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &M0,
                                  const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &M1_T)
{
    std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> retval(m_w_params.rows_M0(),
                                                                              std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(m_w_params.cols_M1()));

    std::exception_ptr p_ex;
    std::mutex mtx_ex;
#pragma omp parallel for collapse(2) num_threads(m_num_threads)
    for (size_t i = 0; i < m_w_params.rows_M0(); ++i)
    {
        for (size_t j = 0; j < m_w_params.cols_M1(); ++j)
        {
            try
            {
                if (!p_ex)
                    retval[i][j] = m_p_context->context()->EvalInnerProduct(M0[i], M1_T[j], pow(2, ceil(log2(m_w_params.cols_M0()))));
            }
            catch (...)
            {
                std::scoped_lock<std::mutex> lock(mtx_ex);
                if (!p_ex)
                    p_ex = std::current_exception();
            }
        } // end for
    } // end for
    if (p_ex)
        std::rethrow_exception(p_ex);
    return retval;
}

//--------------------------
// Provided methods - End
//--------------------------

hebench::APIBridge::Handle MatMultEIPBenchmark::encode(const hebench::APIBridge::PackedData *p_parameters)
{
    std::pair<InternalMatrixPlain, InternalMatrixPlain> params =
        std::make_pair<InternalMatrixPlain, InternalMatrixPlain>(InternalMatrixPlain(0), InternalMatrixPlain(1));

    // encode M0
    //InternalMatrixPlain plain_M0(0);
    InternalMatrixPlain &plain_M0 = params.first;

    const hebench::APIBridge::DataPack &raw_M0 = MatMultEIPBenchmark::findDataPack(*p_parameters, 0);
    if (raw_M0.buffer_count <= 0 || !raw_M0.p_buffers)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid empty data for M0."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    const hebench::APIBridge::NativeDataBuffer &buffer_M0 = *raw_M0.p_buffers;
    std::vector<std::vector<std::int64_t>> matrix_data =
        prepareMatrix(buffer_M0, m_w_params.rows_M0(), m_w_params.cols_M0());
    plain_M0.rows() = encodeM0(matrix_data);

    // encode M1
    //InternalMatrixPlain plain_M1(1);
    InternalMatrixPlain &plain_M1 = params.second;

    const hebench::APIBridge::DataPack &raw_M1 = MatMultEIPBenchmark::findDataPack(*p_parameters, 1);
    if (raw_M1.buffer_count <= 0 || !raw_M1.p_buffers)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid empty data for M1."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    const hebench::APIBridge::NativeDataBuffer &buffer_M1 = *raw_M1.p_buffers;
    matrix_data =
        prepareMatrix(buffer_M1, m_w_params.cols_M0(), m_w_params.cols_M1());
    plain_M1.rows() = encodeM1(matrix_data);

    // wrap our internal object into a handle to cross the boundary of the API Bridge
    return this->getEngine().template createHandle<decltype(params)>(sizeof(lbcrypto::Plaintext) * 2, // size (arbitrary for our usage if we need to)
                                                                     0, // extra tags
                                                                     std::move(params)); // constructor parameters
}

void MatMultEIPBenchmark::decode(hebench::APIBridge::Handle h_encoded_data, hebench::APIBridge::PackedData *p_native)
{
    // able to decode only encoded result

    assert(p_native && p_native->p_data_packs && p_native->pack_count > 0);

    const std::vector<std::vector<lbcrypto::Plaintext>> &encoded_result =
        this->getEngine().template retrieveFromHandle<std::vector<std::vector<lbcrypto::Plaintext>>>(h_encoded_data, MatMultEIPBenchmark::tagEncodedResult);

    if (encoded_result.size() < m_w_params.rows_M0())
    {
        std::stringstream ss;
        ss << "Invalid number of rows in encoded result. Expected " << m_w_params.rows_M0()
           << ", but received " << encoded_result.size() << ".";
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                         HEBENCH_ECODE_INVALID_ARGS);
    } // end if
    for (std::size_t row_i = 0; row_i < m_w_params.rows_M0(); ++row_i)
        if (encoded_result[row_i].size() < m_w_params.cols_M1())
        {
            std::stringstream ss;
            ss << "Invalid number of columns for row " << row_i << " in encoded result. Expected "
               << m_w_params.cols_M1() << ", but received " << encoded_result[row_i].size() << ".";
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                             HEBENCH_ECODE_INVALID_ARGS);
        } // end if

    // index for result component 0
    std::uint64_t data_pack_index = MatMultEIPBenchmark::findDataPackIndex(*p_native, 0);
    if (data_pack_index < p_native->pack_count)
    {
        hebench::APIBridge::DataPack &result_component = p_native->p_data_packs[data_pack_index];
        if (result_component.buffer_count > 0 && result_component.p_buffers)
        {
            hebench::APIBridge::NativeDataBuffer &buffer = result_component.p_buffers[0];
            if (buffer.p && buffer.size > 0)
            {
                // copy as much as we can
                std::int64_t *p_data = reinterpret_cast<std::int64_t *>(buffer.p);
                for (std::size_t row_i = 0; row_i < m_w_params.rows_M0(); ++row_i)
                {
                    for (std::size_t col_i = 0; col_i < m_w_params.cols_M1(); ++col_i)
                    {
                        std::vector<std::int64_t> decoded =
                            encoded_result[row_i][col_i]->GetPackedValue();
                        p_data[m_w_params.cols_M1() * row_i + col_i] = decoded.empty() ?
                                                                         0.0 :
                                                                         decoded.front();
                    } // end for
                } // end for
            } // end if
        } // end if
    } // end if
}

hebench::APIBridge::Handle MatMultEIPBenchmark::encrypt(hebench::APIBridge::Handle h_encoded_data)
{
    const std::pair<InternalMatrixPlain, InternalMatrixPlain> &encoded_params =
        this->getEngine().template retrieveFromHandle<std::pair<InternalMatrixPlain, InternalMatrixPlain>>(h_encoded_data);

    std::pair<InternalMatrixCipher, InternalMatrixCipher> result_encrypted =
        std::make_pair<InternalMatrixCipher, InternalMatrixCipher>(InternalMatrixCipher(encoded_params.first.paramPosition()),
                                                                   InternalMatrixCipher(encoded_params.second.paramPosition()));

    result_encrypted.first.rows()  = encryptMatrix(encoded_params.first.rows());
    result_encrypted.second.rows() = encryptMatrix(encoded_params.second.rows());

    // wrap our internal object into a handle to cross the boundary of the API Bridge
    return this->getEngine().template createHandle<decltype(result_encrypted)>(sizeof(lbcrypto::Ciphertext<lbcrypto::DCRTPoly>) * 2, // size (arbitrary for our usage if we need to)
                                                                               0, // extra tags
                                                                               std::move(result_encrypted)); // constructor parameters
}

hebench::APIBridge::Handle MatMultEIPBenchmark::decrypt(hebench::APIBridge::Handle h_encrypted_data)
{
    // supports only encrypted result
    const std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> &encrypted_result =
        this->getEngine().template retrieveFromHandle<std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>>(h_encrypted_data, MatMultEIPBenchmark::tagEncryptedResult);

    std::vector<std::vector<lbcrypto::Plaintext>> plaintext_result(encrypted_result.size());

    for (size_t i = 0; i < encrypted_result.size(); ++i)
    {
        plaintext_result[i].resize(encrypted_result[i].size());
        for (std::size_t j = 0; j < encrypted_result[i].size(); ++j)
            m_p_context->decrypt(encrypted_result[i][j], plaintext_result[i][j]);
    } // end for

    // wrap our internal object into a handle to cross the boundary of the API Bridge
    return this->getEngine().template createHandle<decltype(plaintext_result)>(plaintext_result.size(),
                                                                               MatMultEIPBenchmark::tagEncodedResult,
                                                                               std::move(plaintext_result));
}

hebench::APIBridge::Handle MatMultEIPBenchmark::load(const hebench::APIBridge::Handle *p_local_data, uint64_t count)
{
    if (count != 1)
        // we do ops in ciphertext only, so we should get 1 pack
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid number of handles. Expected 1."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    // host is same as remote, so, just duplicate handle to let called be able to destroy input handle
    return this->getEngine().duplicateHandle(p_local_data[0]);
}

void MatMultEIPBenchmark::store(hebench::APIBridge::Handle h_remote_data,
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
        p_local_data[0] = this->getEngine().duplicateHandle(h_remote_data, MatMultEIPBenchmark::tagEncryptedResult);
    } // end if
}

hebench::APIBridge::Handle MatMultEIPBenchmark::operate(hebench::APIBridge::Handle h_remote_packed,
                                                        const hebench::APIBridge::ParameterIndexer *p_param_indexers)
{
    const std::pair<InternalMatrixCipher, InternalMatrixCipher> &loaded_data =
        this->getEngine().template retrieveFromHandle<std::pair<InternalMatrixCipher, InternalMatrixCipher>>(h_remote_packed);

    assert(loaded_data.first.paramPosition() == 0
           && loaded_data.second.paramPosition() == 1);

    // validate indexers
    for (std::size_t param_i = 0; param_i < ParametersCount; ++param_i)
    {
        if (p_param_indexers[param_i].value_index > 0 || p_param_indexers[param_i].batch_size != 1)
        {
            std::stringstream ss;
            ss << "Invalid parameter indexer for operation parameter " << param_i << ". Expected index in range [0, 1), but ["
               << p_param_indexers[param_i].value_index << ", " << p_param_indexers[param_i].batch_size << ") received.";
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                             HEBENCH_ECODE_INVALID_ARGS);
        } // end if
    } // end for

    std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> retval =
        doMatMultEIP(loaded_data.first.rows(), loaded_data.second.rows());

    // send our internal result across the boundary of the API Bridge as a handle
    return this->getEngine().template createHandle<decltype(retval)>(sizeof(retval),
                                                                     MatMultEIPBenchmark::tagEncryptedResult,
                                                                     std::move(retval));
}

} // namespace bfv
} // namespace pbe
