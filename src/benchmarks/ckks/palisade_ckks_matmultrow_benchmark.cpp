
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

#include "benchmarks/ckks/palisade_ckks_matmultrow_benchmark.h"
#include "engine/palisade_engine.h"
#include "engine/palisade_error.h"

#include <omp.h>

namespace pbe {
namespace ckks {

//-----------------------------------
// class MatMultRowBenchmarkDescription
//-----------------------------------

MatMultRowBenchmarkDescription::MatMultRowBenchmarkDescription()
{
    // initialize the descriptor for this benchmark
    std::memset(&m_descriptor, 0, sizeof(hebench::APIBridge::BenchmarkDescriptor));
    m_descriptor.workload          = hebench::APIBridge::Workload::MatrixMultiply;
    m_descriptor.data_type         = hebench::APIBridge::DataType::Float64;
    m_descriptor.category          = hebench::APIBridge::Category::Latency;
    m_descriptor.cipher_param_mask = HEBENCH_HE_PARAM_FLAGS_ALL_CIPHER;
    //
    m_descriptor.cat_params.min_test_time_ms                = 0;
    m_descriptor.cat_params.latency.warmup_iterations_count = 1;
    //
    m_descriptor.scheme   = HEBENCH_HE_SCHEME_CKKS;
    m_descriptor.security = HEBPALISADE_HE_SECURITY_128;
    m_descriptor.other    = MatMultRowID;

    // specify default arguments for this workload:
    hebench::cpp::WorkloadParams::MatrixMultiply default_workload_params;
    default_workload_params.rows_M0() = 10;
    default_workload_params.cols_M0() = 9;
    default_workload_params.cols_M1() = 8;
    default_workload_params.add<std::uint64_t>(MatMultRowBenchmarkDescription::DefaultPolyModulusDegree, "PolyModulusDegree");
    default_workload_params.add<std::uint64_t>(MatMultRowBenchmarkDescription::DefaultNumCoefficientModuli, "MultiplicativeDepth");
    default_workload_params.add<std::uint64_t>(MatMultRowBenchmarkDescription::DefaultScaleExponent, "ScaleBits");
    default_workload_params.add<std::uint64_t>(MatMultRowBenchmarkDescription::DefaultNumThreads, "NumThreads");
    // total: 7 workload params
    this->addDefaultParameters(default_workload_params);
}

MatMultRowBenchmarkDescription::~MatMultRowBenchmarkDescription()
{
    //
}

std::string MatMultRowBenchmarkDescription::getBenchmarkDescription(const hebench::APIBridge::WorkloadParams *p_w_params) const
{
    std::stringstream ss;
    std::string s_tmp = BenchmarkDescription::getBenchmarkDescription(p_w_params);

    if (!p_w_params)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid null workload parameters `p_w_params`"),
                                         HEBENCH_ECODE_INVALID_ARGS);

    assert(p_w_params->count >= MatMultRowBenchmarkDescription::NumWorkloadParams);

    std::size_t pmd           = p_w_params->params[Index_PolyModulusDegree].u_param;
    std::size_t mult_depth    = p_w_params->params[Index_NumCoefficientModuli].u_param;
    std::size_t scale_bits    = p_w_params->params[Index_ScaleExponent].u_param;
    std::uint64_t num_threads = p_w_params->params[MatMultRowBenchmarkDescription::Index_NumThreads].u_param;

    if (num_threads <= 0)
        num_threads = omp_get_max_threads();
    if (!s_tmp.empty())
        ss << s_tmp << std::endl;
    ss << ", Encryption parameters" << std::endl
       << ", , HE Library, PALISADE 1.11.3" << std::endl
       << ", , Key-switching technique, PALISADE Hybrid" << std::endl
       << ", , Poly modulus degree, " << pmd << std::endl
       << ", , Multiplicative Depth, " << mult_depth << std::endl
       << ", , Scale, 2^" << scale_bits << std::endl
       << ", Algorithm, " << AlgorithmName << ", " << AlgorithmDescription << std::endl
       << ", Number of threads, " << num_threads;
    return ss.str();
}

hebench::cpp::BaseBenchmark *MatMultRowBenchmarkDescription::createBenchmark(hebench::cpp::BaseEngine &engine,
                                                                             const hebench::APIBridge::WorkloadParams *p_params)
{
    if (!p_params)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid empty workload parameters. This workload requires flexible parameters."),
                                         HEBENCH_ECODE_CRITICAL_ERROR);

    PalisadeEngine &ex_engine = dynamic_cast<PalisadeEngine &>(engine);
    return new MatMultRowBenchmark(ex_engine, m_descriptor, *p_params);
}

void MatMultRowBenchmarkDescription::destroyBenchmark(hebench::cpp::BaseBenchmark *p_bench)
{
    MatMultRowBenchmark *p = dynamic_cast<MatMultRowBenchmark *>(p_bench);
    if (p)
        delete p;
}

//------------------------
// class MatMultRowBenchmark
//------------------------

MatMultRowBenchmark::MatMultRowBenchmark(PalisadeEngine &engine,
                                         const hebench::APIBridge::BenchmarkDescriptor &bench_desc,
                                         const hebench::APIBridge::WorkloadParams &bench_params) :
    hebench::cpp::BaseBenchmark(engine, bench_desc, bench_params),
    m_w_params(bench_params)
{
    // validate workload parameters

    // number of workload parameters (3 for matmult; +3 encryption params)
    if (bench_params.count < MatMultRowBenchmarkDescription::NumWorkloadParams)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid workload parameters. This workload requires " + std::to_string(MatMultRowBenchmarkDescription::NumWorkloadParams) + " parameters."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    std::size_t pmd        = m_w_params.get<std::uint64_t>(MatMultRowBenchmarkDescription::Index_PolyModulusDegree);
    std::size_t mult_depth = m_w_params.get<std::uint64_t>(MatMultRowBenchmarkDescription::Index_NumCoefficientModuli);
    std::size_t scale_bits = m_w_params.get<std::uint64_t>(MatMultRowBenchmarkDescription::Index_ScaleExponent);
    m_num_threads          = static_cast<int>(m_w_params.get<std::uint64_t>(MatMultRowBenchmarkDescription::Index_NumThreads));
    if (m_num_threads <= 0)
        m_num_threads = omp_get_max_threads();

    // check values of the workload parameters and make sure they are supported by benchmark:

    if (m_w_params.rows_M0() <= 0 || m_w_params.cols_M0() <= 0 || m_w_params.cols_M1() <= 0)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Matrix dimensions must be greater than 0."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    if (m_w_params.cols_M0() > (pmd / 2) || m_w_params.cols_M0() * m_w_params.cols_M1() > (pmd / 2))
    {
        std::stringstream ss;
        ss << "Invalid workload parameters. This workload only supports matrices of dimensions (a x b) x (b x c) where 'b' and b * c is at max " << (pmd / 2) << " (e.g. PolyModulusDegree / 2).";
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                         HEBENCH_ECODE_INVALID_ARGS);
    } // end if

    int n              = 0;
    int spacers        = 0;
    m_encoding_spacers = pmd / 2 / m_w_params.cols_M0();
    spacers            = m_encoding_spacers;
    std::vector<int32_t> vec(m_w_params.cols_M0());
    std::generate(std::begin(vec), std::end(vec), [&n, &spacers] { return n += spacers; });

    m_p_context = PalisadeContext::createCKKSContext(pmd, mult_depth, scale_bits, m_w_params.cols_M0());
    m_p_context->EvalMultKeyGen();
    m_p_context->EvalAtIndexKeyGen(vec);
    m_p_context->EvalSumKeyGen();
}

MatMultRowBenchmark::~MatMultRowBenchmark()
{
    // nothing needed in this example
}

//--------------------------
// Provided methods - Start
//--------------------------

std::vector<std::vector<double>> MatMultRowBenchmark::prepareMatrix(const hebench::APIBridge::NativeDataBuffer &buffer,
                                                                    std::uint64_t rows, std::uint64_t cols)
{
    std::vector<std::vector<double>> retval(rows, std::vector<double>(cols));
    if (!buffer.p || buffer.size < rows * cols * sizeof(double))
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Insufficient data for M0."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    const double *p_curr_row = reinterpret_cast<const double *>(buffer.p);
    for (std::size_t row_i = 0; row_i < rows; ++row_i)
    {
        std::copy(p_curr_row, p_curr_row + cols, retval[row_i].begin());
        p_curr_row += cols;
    } // end for
    return retval;
}

std::vector<lbcrypto::Plaintext> MatMultRowBenchmark::encodeM0(const std::vector<std::vector<double>> &data)
{
    assert(data.size() == m_w_params.rows_M0());
    assert(!data.empty() && data.front().size() == m_w_params.cols_M0());

    std::vector<double> vec_a(m_p_context->getSlotCount(), 0);
    std::vector<std::vector<double>> vec_container_a(m_w_params.rows_M0());

    for (size_t i = 0; i < m_w_params.rows_M0(); i++)
    {
        for (size_t j = 0; j < m_w_params.cols_M0(); j++)
        {
            for (size_t k = 0; k < m_w_params.cols_M1(); k++)
            {
                vec_a[m_encoding_spacers * j + k] = static_cast<double>(data[i][j]);
            }
        }
        vec_container_a[i] = vec_a;
    }

    std::vector<lbcrypto::Plaintext> plainM0;

    for (size_t i = 0; i < vec_container_a.size(); i++)
        plainM0.push_back(m_p_context->context()->MakeCKKSPackedPlaintext(vec_container_a[i]));
    return plainM0;
}

std::vector<lbcrypto::Plaintext> MatMultRowBenchmark::encodeM1(const std::vector<std::vector<double>> &data)
{
    assert(data.size() == m_w_params.cols_M0());
    assert(!data.empty() && data.front().size() == m_w_params.cols_M1());

    std::vector<double> vec_b(m_p_context->getSlotCount(), 0);
    std::vector<std::vector<double>> vec_container_b(1);

    for (size_t j = 0; j < m_w_params.cols_M0(); j++)
    {
        for (size_t k = 0; k < m_w_params.cols_M1(); k++)
        {
            vec_b[m_encoding_spacers * j + k] = static_cast<double>(data[j][k]);
        }
    }
    vec_container_b[0] = vec_b;

    std::vector<lbcrypto::Plaintext> plainM1;

    for (size_t i = 0; i < vec_container_b.size(); i++)
        plainM1.push_back(m_p_context->context()->MakeCKKSPackedPlaintext(vec_container_b[i]));

    return plainM1;
}

std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> MatMultRowBenchmark::encryptMatrix(const std::vector<lbcrypto::Plaintext> &plain)
{
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> retval(plain.size());
    for (size_t i = 0; i < plain.size(); ++i)
    {
        retval[i] = m_p_context->context()->Encrypt(m_p_context->publicKey(), plain[i]);
    } // end for
    return retval;
}

std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>
MatMultRowBenchmark::doMatMultRow(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &M0,
                                  const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &M1)
{
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> retval(m_w_params.rows_M0());

    lbcrypto::Plaintext plain_zero                       = m_p_context->context()->MakeCKKSPackedPlaintext(std::vector<double>{ 0.0 });
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cipher_zero = m_p_context->context()->Encrypt(m_p_context->publicKey(), plain_zero);
    plain_zero.reset();

    int num_threads = m_num_threads;
    int threads_at_level[2];
    threads_at_level[0] = static_cast<int>(M0.size());
    if (threads_at_level[0] > num_threads)
        threads_at_level[0] = num_threads;
    threads_at_level[1] = num_threads / threads_at_level[0];
    if (threads_at_level[1] < 1)
        threads_at_level[1] = 1;

    const int old_max_active_levels = omp_get_max_active_levels();
    const int old_nested_value      = omp_get_nested();
    omp_set_nested(true);
    omp_set_max_active_levels(2);

#pragma omp parallel for num_threads(threads_at_level[0])
    for (size_t i = 0; i < M0.size(); i++)
    {
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cipher_res_tmp = m_p_context->context()->EvalMult(M0[i], M1[0]);
        auto cPrecomp                                           = m_p_context->context()->EvalFastRotationPrecompute(cipher_res_tmp);
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly> tmp_sum        = cipher_zero;
        //m_p_context->context()->RescaleInPlace(cipher_res_tmp);
        tmp_sum->SetDepth(cipher_res_tmp->GetDepth());

#pragma omp declare reduction(+                                           \
                              : lbcrypto::Ciphertext <lbcrypto::DCRTPoly> \
                              : omp_out += omp_in)                        \
    initializer(omp_priv = omp_orig)
#pragma omp parallel for reduction(+ \
                                   : tmp_sum) num_threads(threads_at_level[1])
        for (int32_t k = 1; k < m_w_params.cols_M0(); k++)
        {
            tmp_sum += m_p_context->context()->EvalFastRotation(cipher_res_tmp, k * m_encoding_spacers, m_p_context->getSlotCount() * 4, cPrecomp);
        }
        retval[i] = tmp_sum + cipher_res_tmp;
    }

    omp_set_max_active_levels(old_max_active_levels);
    omp_set_nested(old_nested_value);

    return retval;
}

//--------------------------
// Provided methods - End
//--------------------------

hebench::APIBridge::Handle MatMultRowBenchmark::encode(const hebench::APIBridge::DataPackCollection *p_parameters)
{
    std::pair<InternalMatrixPlain, InternalMatrixPlain> params =
        std::make_pair<InternalMatrixPlain, InternalMatrixPlain>(InternalMatrixPlain(0), InternalMatrixPlain(1));

    // encode M0
    //InternalMatrixPlain plain_M0(0);
    InternalMatrixPlain &plain_M0 = params.first;

    const hebench::APIBridge::DataPack &raw_M0 = MatMultRowBenchmark::findDataPack(*p_parameters, 0);
    if (raw_M0.buffer_count <= 0 || !raw_M0.p_buffers)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid empty data for M0."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    const hebench::APIBridge::NativeDataBuffer &buffer_M0 = *raw_M0.p_buffers;

    std::vector<std::vector<double>> matrix_data =
        prepareMatrix(buffer_M0, m_w_params.rows_M0(), m_w_params.cols_M0());
    plain_M0.rows() = encodeM0(matrix_data);

    // encode M1
    //InternalMatrixPlain plain_M1(1);
    InternalMatrixPlain &plain_M1 = params.second;

    const hebench::APIBridge::DataPack &raw_M1 = MatMultRowBenchmark::findDataPack(*p_parameters, 1);
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

void MatMultRowBenchmark::decode(hebench::APIBridge::Handle h_encoded_data, hebench::APIBridge::DataPackCollection *p_native)
{
    // able to decode only encoded result

    assert(p_native && p_native->p_data_packs && p_native->pack_count > 0);

    const std::vector<lbcrypto::Plaintext> &encoded_result =
        this->getEngine().template retrieveFromHandle<std::vector<lbcrypto::Plaintext>>(h_encoded_data, MatMultRowBenchmark::tagEncodedResult);

    if (encoded_result.size() < m_w_params.rows_M0())
    {
        std::stringstream ss;
        ss << "Invalid number of rows in encoded result. Expected " << m_w_params.rows_M0()
           << ", but received " << encoded_result.size() << ".";
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                         HEBENCH_ECODE_INVALID_ARGS);
    } // end if

    // index for result component 0
    std::uint64_t data_pack_index = MatMultRowBenchmark::findDataPackIndex(*p_native, 0);
    if (data_pack_index < p_native->pack_count)
    {
        hebench::APIBridge::DataPack &result_component = p_native->p_data_packs[data_pack_index];
        if (result_component.buffer_count > 0 && result_component.p_buffers)
        {
            hebench::APIBridge::NativeDataBuffer &buffer = result_component.p_buffers[0];
            if (buffer.p && buffer.size > 0)
            {
                // copy as much as we can
                double *p_data = reinterpret_cast<double *>(buffer.p);
                for (std::size_t row_i = 0; row_i < m_w_params.rows_M0(); ++row_i)
                {
                    std::vector<double> decoded = encoded_result[row_i]->GetRealPackedValue();
                    decoded.resize(m_w_params.cols_M1());
                    std::vector<double> test(m_w_params.cols_M0(), 0);
                    if (decoded.empty())
                        std::copy(test.begin(), test.end(), p_data + (m_w_params.cols_M1() * row_i)); // p_data[m_w_params.cols_M1 * row_i] = test.data();
                    else
                    {
                        for (size_t i = 0; i < decoded.size(); i++)
                            if (std::abs(decoded[i]) < 0.00005)
                                decoded[i] = 0.0;
                        std::copy(decoded.begin(), decoded.end(), p_data + (m_w_params.cols_M1() * row_i)); //p_data[m_w_params.cols_M1 * row_i] = decoded.data();
                    } // end if
                } // end for
            } // end if
        } // end if
    } // end if
}

hebench::APIBridge::Handle MatMultRowBenchmark::encrypt(hebench::APIBridge::Handle h_encoded_data)
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

hebench::APIBridge::Handle MatMultRowBenchmark::decrypt(hebench::APIBridge::Handle h_encrypted_data)
{
    // supports only encrypted result
    const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &encrypted_result =
        this->getEngine().template retrieveFromHandle<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>(h_encrypted_data, MatMultRowBenchmark::tagEncryptedResult);

    std::vector<lbcrypto::Plaintext> plaintext_result(encrypted_result.size());

    for (size_t i = 0; i < encrypted_result.size(); ++i)
    {
        m_p_context->decrypt(encrypted_result[i], plaintext_result[i]);
    } // end for

    // wrap our internal object into a handle to cross the boundary of the API Bridge
    return this->getEngine().template createHandle<decltype(plaintext_result)>(plaintext_result.size(),
                                                                               MatMultRowBenchmark::tagEncodedResult,
                                                                               std::move(plaintext_result));
}

hebench::APIBridge::Handle MatMultRowBenchmark::load(const hebench::APIBridge::Handle *p_local_data, uint64_t count)
{
    if (count != 1)
        // we do ops in ciphertext only, so we should get 1 pack
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid number of handles. Expected 1."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    // host is same as remote, so, just duplicate handle to let called be able to destroy input handle
    return this->getEngine().duplicateHandle(p_local_data[0]);
}

void MatMultRowBenchmark::store(hebench::APIBridge::Handle h_remote_data,
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
        p_local_data[0] = this->getEngine().duplicateHandle(h_remote_data, MatMultRowBenchmark::tagEncryptedResult);
    } // end if
}

hebench::APIBridge::Handle MatMultRowBenchmark::operate(hebench::APIBridge::Handle h_remote_packed,
                                                        const hebench::APIBridge::ParameterIndexer *p_param_indexers,
                                                        std::uint64_t indexers_count)
{
    if (indexers_count < MatMultRowBenchmarkDescription::NumOpParams)
    {
        std::stringstream ss;
        ss << "Invalid number of indexers. Expected " << MatMultRowBenchmarkDescription::NumOpParams
           << ", but " << indexers_count << " received." << std::endl;
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()), HEBENCH_ECODE_INVALID_ARGS);
    } // end if
    const std::pair<InternalMatrixCipher, InternalMatrixCipher> &loaded_data =
        this->getEngine().template retrieveFromHandle<std::pair<InternalMatrixCipher, InternalMatrixCipher>>(h_remote_packed);

    assert(loaded_data.first.paramPosition() == 0
           && loaded_data.second.paramPosition() == 1);

    // validate indexers
    for (std::size_t param_i = 0; param_i < MatMultRowBenchmarkDescription::NumOpParams; ++param_i)
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

    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> retval =
        doMatMultRow(loaded_data.first.rows(), loaded_data.second.rows());

    // send our internal result across the boundary of the API Bridge as a handle
    return this->getEngine().template createHandle<decltype(retval)>(sizeof(retval),
                                                                     MatMultRowBenchmark::tagEncryptedResult,
                                                                     std::move(retval));
}

} // namespace ckks
} // namespace pbe
