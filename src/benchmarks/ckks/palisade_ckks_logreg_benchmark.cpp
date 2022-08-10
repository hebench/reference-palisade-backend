
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

#include "benchmarks/ckks/palisade_ckks_logreg_benchmark.h"
#include "engine/palisade_engine.h"
#include "engine/palisade_error.h"
#include <omp.h>

namespace pbe {
namespace ckks {

const std::vector<double> LogRegBenchmark::SigmoidPolyCoeff = { 0.5, 0.15012, 0.0, -0.0015930078125 };

//-----------------------------------
// class LogRegBenchmarkDescription
//-----------------------------------

LogRegBenchmarkDescription::LogRegBenchmarkDescription(hebench::APIBridge::Category category, std::size_t batch_size)
{
    // initialize the descriptor for this benchmark
    std::memset(&m_descriptor, 0, sizeof(hebench::APIBridge::BenchmarkDescriptor));
    m_descriptor.data_type = hebench::APIBridge::DataType::Float64;
    m_descriptor.category  = category;
    switch (category)
    {
    case hebench::APIBridge::Category::Latency:
        m_descriptor.cat_params.min_test_time_ms                = 0; // read from user
        m_descriptor.cat_params.latency.warmup_iterations_count = 1;
        break;

    case hebench::APIBridge::Category::Offline:
        m_descriptor.cat_params.offline.data_count[Index_W] = 1;
        m_descriptor.cat_params.offline.data_count[Index_b] = 1;
        m_descriptor.cat_params.offline.data_count[Index_X] = batch_size;
        break;

    default:
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid category received."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    }
    m_descriptor.cipher_param_mask = HEBENCH_HE_PARAM_FLAGS_ALL_CIPHER;
    //
    m_descriptor.scheme   = HEBENCH_HE_SCHEME_CKKS;
    m_descriptor.security = HEBPALISADE_HE_SECURITY_128;
    m_descriptor.other    = LogRegOtherID;
    m_descriptor.workload = hebench::APIBridge::Workload::LogisticRegression_PolyD3;

    hebench::cpp::WorkloadParams::LogisticRegression default_workload_params;
    default_workload_params.n() = 16;
    default_workload_params.add<std::uint64_t>(LogRegBenchmarkDescription::DefaultPolyModulusDegree, "PolyModulusDegree");
    default_workload_params.add<std::uint64_t>(LogRegBenchmarkDescription::DefaultNumCoefficientModuli, "MultiplicativeDepth");
    default_workload_params.add<std::uint64_t>(LogRegBenchmarkDescription::DefaultScaleExponent, "ScaleBits");
    default_workload_params.add<std::uint64_t>(LogRegBenchmarkDescription::DefaultNumThreads, "NumThreads");
    this->addDefaultParameters(default_workload_params);
}

LogRegBenchmarkDescription::~LogRegBenchmarkDescription()
{
    //
}

std::string LogRegBenchmarkDescription::getBenchmarkDescription(const hebench::APIBridge::WorkloadParams *p_w_params) const
{
    std::stringstream ss;
    std::string s_tmp = BenchmarkDescription::getBenchmarkDescription(p_w_params);

    if (!p_w_params)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid null workload parameters `p_w_params`"),
                                         HEBENCH_ECODE_INVALID_ARGS);

    assert(p_w_params->count >= LogRegBenchmarkDescription::NumWorkloadParams);

    std::size_t pmd           = p_w_params->params[Index_PolyModulusDegree].u_param;
    std::size_t mult_depth    = p_w_params->params[Index_NumCoefficientModuli].u_param;
    std::size_t scale_bits    = p_w_params->params[Index_ScaleExponent].u_param;
    std::uint64_t num_threads = p_w_params->params[Index_NumThreads].u_param;
    if (num_threads <= 0)
        num_threads = omp_get_max_threads();

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

hebench::cpp::BaseBenchmark *LogRegBenchmarkDescription::createBenchmark(hebench::cpp::BaseEngine &engine,
                                                                         const hebench::APIBridge::WorkloadParams *p_params)
{
    if (!p_params)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid empty workload parameters. This workload requires flexible parameters."),
                                         HEBENCH_ECODE_CRITICAL_ERROR);

    PalisadeEngine &ex_engine = dynamic_cast<PalisadeEngine &>(engine);
    return new LogRegBenchmark(ex_engine, m_descriptor, *p_params);
}

void LogRegBenchmarkDescription::destroyBenchmark(hebench::cpp::BaseBenchmark *p_bench)
{
    LogRegBenchmark *p = dynamic_cast<LogRegBenchmark *>(p_bench);
    if (p)
        delete p;
}

//------------------------
// class LogRegBenchmark
//------------------------
#include <iostream>
LogRegBenchmark::LogRegBenchmark(PalisadeEngine &engine,
                                 const hebench::APIBridge::BenchmarkDescriptor &bench_desc,
                                 const hebench::APIBridge::WorkloadParams &bench_params) :
    hebench::cpp::BaseBenchmark(engine, bench_desc, bench_params),
    m_sample_size(0),
    m_w_params(bench_params)
{
    // validate workload parameters

    const hebench::APIBridge::BenchmarkDescriptor &local_bench_desc = getDescriptor();

    if (local_bench_desc.workload != hebench::APIBridge::Workload::LogisticRegression_PolyD3
        || local_bench_desc.data_type != hebench::APIBridge::DataType::Float64
        || (local_bench_desc.category != hebench::APIBridge::Category::Latency
            && local_bench_desc.category != hebench::APIBridge::Category::Offline)
        || ((local_bench_desc.cipher_param_mask & 0x03) != 0x03)
        || local_bench_desc.scheme != HEBENCH_HE_SCHEME_CKKS
        || local_bench_desc.security != HEBPALISADE_HE_SECURITY_128
        || local_bench_desc.other != LogRegBenchmarkDescription::LogRegOtherID)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Benchmark descriptor received is not supported."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    if (local_bench_desc.category == hebench::APIBridge::Category::Offline
        && (local_bench_desc.cat_params.offline.data_count[LogRegBenchmarkDescription::Index_W] > 1
            || local_bench_desc.cat_params.offline.data_count[LogRegBenchmarkDescription::Index_b] > 1))
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Benchmark descriptor received is not supported."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    std::size_t pmd        = m_w_params.get<std::uint64_t>(LogRegBenchmarkDescription::Index_PolyModulusDegree);
    std::size_t mult_depth = m_w_params.get<std::uint64_t>(LogRegBenchmarkDescription::Index_NumCoefficientModuli);
    std::size_t scale_bits = m_w_params.get<std::uint64_t>(LogRegBenchmarkDescription::Index_ScaleExponent);
    m_num_threads          = static_cast<int>(m_w_params.get<std::uint64_t>(LogRegBenchmarkDescription::Index_NumThreads));
    if (m_num_threads <= 0)
        m_num_threads = omp_get_max_threads();

    // check values of the workload parameters and make sure they are supported by benchmark:

    if (m_w_params.n() > pmd / 2)
    {
        std::stringstream ss;
        ss << "Invalid workload parameters. This workload only supports feature vectors of size less than "
           << pmd / 2 << ", but " << m_w_params.n() << " specified.";
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                         HEBENCH_ECODE_INVALID_ARGS);
    } // end if

    m_p_context = PalisadeContext::createCKKSContext(pmd, mult_depth, scale_bits,
                                                     lbcrypto::HEStd_128_classic, m_w_params.n());
    m_p_context->EvalMultKeyGen();
    m_p_context->EvalSumKeyGen();
    // Rotation Keys will be generated when knowledge of batch size is passed from remote
}

LogRegBenchmark::~LogRegBenchmark()
{
    // nothing needed in this example
}

//--------------------------
// Provided methods - Start
//--------------------------

std::vector<std::vector<double>> LogRegBenchmark::prepareData(const hebench::APIBridge::DataPack &data_pack)
{
    if (!data_pack.p_buffers || data_pack.buffer_count <= 0)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid empty data pack for operation parameter " + std::to_string(data_pack.param_position) + "."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    std::vector<std::vector<double>> retval(data_pack.buffer_count);
    for (std::size_t sample_i = 0; sample_i < data_pack.buffer_count; ++sample_i)
    {
        hebench::APIBridge::NativeDataBuffer &buffer = data_pack.p_buffers[sample_i];
        const double *buffer_begin                   = reinterpret_cast<const double *>(buffer.p);
        const double *buffer_end                     = buffer_begin + buffer.size / sizeof(double);
        if (buffer_begin)
            retval[sample_i].assign(buffer_begin, buffer_end);
    } // end for

    return retval;
}

std::vector<std::vector<double>> LogRegBenchmark::prepareInputs(const hebench::APIBridge::DataPack &data_pack)
{
    std::vector<std::vector<double>> retval = prepareData(data_pack);

    m_sample_size = retval.size();
    // validate sample size
    const hebench::APIBridge::BenchmarkDescriptor &local_bench_desc = getDescriptor();
    if (local_bench_desc.category == hebench::APIBridge::Category::Latency)
    {
        if (m_sample_size != 1)
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid number of samples for latency category. Expected 1 sample, but received " + std::to_string(m_sample_size) + "."),
                                             HEBENCH_ECODE_INVALID_ARGS);
    } // end if
    else if (local_bench_desc.category == hebench::APIBridge::Category::Offline)
    {
        std::uint64_t cat_param_sample_size = local_bench_desc.cat_params.offline.data_count[LogRegBenchmarkDescription::Index_X];
        if (cat_param_sample_size > 0)
        {
            if (m_sample_size != cat_param_sample_size)
                throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid number of samples for offline category. Expected " + std::to_string(cat_param_sample_size) + " samples, but received " + std::to_string(m_sample_size) + "."),
                                                 HEBENCH_ECODE_INVALID_ARGS);
        } // end if
    } // end else if

    // Generating Rotation Keys here as backend should now have knowledge of batch size passed from config or documentation
    std::vector<std::int32_t> il(m_sample_size);
    for (std::uint32_t i = 0; i < il.size(); ++i)
        il[i] = -1 - i;
    m_p_context->EvalAtIndexKeyGen(il);

    return retval;
}

lbcrypto::Plaintext LogRegBenchmark::encodeWeightsFromDataPack(const hebench::APIBridge::DataPack &data_pack)
{
    if (data_pack.buffer_count < 1)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Insufficient data for weights."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    return encodeVector(prepareData(data_pack).front());
}

lbcrypto::Plaintext LogRegBenchmark::encodeBiasFromDataPack(const hebench::APIBridge::DataPack &data_pack)
{
    if (!data_pack.p_buffers || data_pack.buffer_count < 1)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Insufficient data for bias."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    const hebench::APIBridge::NativeDataBuffer &buffer = data_pack.p_buffers[0];
    const double *p_bias                               = reinterpret_cast<const double *>(buffer.p);
    if (!p_bias || buffer.size < sizeof(double))
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Insufficient data for bias in native buffer."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    return encodeBias(*p_bias);
}

std::vector<std::vector<lbcrypto::Plaintext>> LogRegBenchmark::encodeInputsFromDataPack(const hebench::APIBridge::DataPack &data_pack)
{
    if (!data_pack.p_buffers || data_pack.buffer_count < 1)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid empty input data."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    return encodeInputs(prepareInputs(data_pack));
}

lbcrypto::Plaintext LogRegBenchmark::encodeVector(const std::vector<double> &data)
{
    return m_p_context->context()->MakeCKKSPackedPlaintext(data);
}

lbcrypto::Plaintext LogRegBenchmark::encodeBias(double data)
{
    return encodeVector(std::vector<double>(m_p_context->getSlotCount(), data));
}

std::vector<std::vector<lbcrypto::Plaintext>> LogRegBenchmark::encodeInputs(const std::vector<std::vector<double>> &data)
{
    std::vector<std::vector<lbcrypto::Plaintext>> retval;

    std::size_t num_batches = data.size() / m_p_context->getSlotCount()
                              + (data.size() % m_p_context->getSlotCount() != 0 ? 1 : 0);
    retval.resize(num_batches);
    std::size_t data_sample_i = 0;
    for (std::size_t batch_i = 0; batch_i < retval.size(); ++batch_i)
    {
        std::vector<lbcrypto::Plaintext> &batch = retval[batch_i];
        std::size_t batch_size                  = data.size() - data_sample_i >= m_p_context->getSlotCount() ?
                                     m_p_context->getSlotCount() :
                                     data.size() - data_sample_i;
        batch.resize(batch_size);
        for (std::size_t batch_sample_i = 0; batch_sample_i < batch.size(); ++batch_sample_i)
        {
            batch[batch_sample_i] = encodeVector(data[data_sample_i]);
            ++data_sample_i;
        } // end for
    } // end for
    return retval;
}

std::vector<double> LogRegBenchmark::decodeResult(const std::vector<lbcrypto::Plaintext> &pt_result, std::size_t n_samples) const
{
    std::size_t n_batches  = pt_result.size();
    std::size_t slot_count = m_p_context->getSlotCount();

    std::vector<double> retval(n_batches * slot_count);

    for (std::size_t i = 0; i < n_batches; ++i)
    {
        std::vector<double> buf = pt_result[i]->GetRealPackedValue();
        std::copy_n(buf.begin(), slot_count, &retval[i * slot_count]);
    }

    retval.resize(n_samples);
#pragma omp parallel for num_threads(m_num_threads)
    for (std::size_t i = 0; i < retval.size(); ++i)
        if (std::abs(retval[i]) < 0.00005)
            retval[i] = 0;
    return retval;
}

std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>
LogRegBenchmark::encryptInputs(const std::vector<std::vector<lbcrypto::Plaintext>> &encoded_inputs)
{
    std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> retval(encoded_inputs.size());

    for (std::size_t batch_i = 0; batch_i < retval.size(); ++batch_i)
    {
        retval[batch_i].resize(encoded_inputs[batch_i].size());
        for (std::size_t sample_i = 0; sample_i < retval[batch_i].size(); ++sample_i)
        {
            retval[batch_i][sample_i] = m_p_context->context()->Encrypt(m_p_context->publicKey(),
                                                                        encoded_inputs[batch_i][sample_i]);
        } // end for
    } // end for

    return retval;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
LogRegBenchmark::doLogReg(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &w,
                          const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &bias,
                          const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &inputs,
                          std::size_t weights_count,
                          int threads)
{
    std::exception_ptr p_ex;
    std::mutex mtx_ex;

    // linear regression
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> dots(inputs.size());
#pragma omp parallel for num_threads(threads)
    for (std::size_t sample_i = 0; sample_i < inputs.size(); ++sample_i)
    {
        try
        {
            if (!p_ex)
                dots[sample_i] = m_p_context->context()->EvalInnerProduct(w, inputs[sample_i], weights_count);
        }
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(mtx_ex);
            if (!p_ex)
                p_ex = std::current_exception();
        }
    } // end for
    if (p_ex)
        std::rethrow_exception(p_ex);
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> retval = m_p_context->context()->EvalMerge(dots);
    m_p_context->context()->RescaleInPlace(retval);
    m_p_context->context()->RescaleInPlace(retval);
    retval = m_p_context->context()->EvalAdd(retval, bias);
    // sigmoid
    retval = m_p_context->context()->EvalPoly(retval, SigmoidPolyCoeff);
    return retval;
}

//--------------------------
// Provided methods - End
//--------------------------

hebench::APIBridge::Handle LogRegBenchmark::encode(const hebench::APIBridge::DataPackCollection *p_parameters)
{
    if (p_parameters->pack_count != LogRegBenchmarkDescription::OpParamsCount)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid number of operation parameters detected in parameter pack. Expected "
                                                            + std::to_string(LogRegBenchmarkDescription::OpParamsCount) + "."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    // validate all op parameters are in this pack
    for (std::uint64_t param_i = 0; param_i < LogRegBenchmarkDescription::OpParamsCount; ++param_i)
    {
        if (findDataPackIndex(*p_parameters, param_i) >= p_parameters->pack_count)
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("DataPack for Logistic Regression inference operation parameter " + std::to_string(param_i) + " expected, but not found in 'p_parameters'."),
                                             HEBENCH_ECODE_INVALID_ARGS);
    } // end for

    const hebench::APIBridge::DataPack &pack_W = findDataPack(*p_parameters, LogRegBenchmarkDescription::Index_W);
    const hebench::APIBridge::DataPack &pack_b = findDataPack(*p_parameters, LogRegBenchmarkDescription::Index_b);
    const hebench::APIBridge::DataPack &pack_X = findDataPack(*p_parameters, LogRegBenchmarkDescription::Index_X);

    return this->getEngine().createHandle<EncodedOpParams>(sizeof(EncodedOpParams),
                                                           tagEncodedOpParams,
                                                           std::make_tuple(encodeWeightsFromDataPack(pack_W),
                                                                           encodeBiasFromDataPack(pack_b),
                                                                           encodeInputsFromDataPack(pack_X)));
}

void LogRegBenchmark::decode(hebench::APIBridge::Handle h_encoded_data, hebench::APIBridge::DataPackCollection *p_native)
{
    // able to decode only encoded result

    assert(p_native && p_native->p_data_packs && p_native->pack_count > 0);

    // index for result component 0
    std::uint64_t data_pack_index = LogRegBenchmark::findDataPackIndex(*p_native, 0);
    if (data_pack_index < p_native->pack_count)
    {
        // find minimum batch size to decode
        hebench::APIBridge::DataPack &result = p_native->p_data_packs[0];
        std::uint64_t min_count              = std::min(result.buffer_count, m_sample_size);
        if (min_count > 0)
        {
            // decode into local format
            const std::vector<lbcrypto::Plaintext> &encoded_result =
                this->getEngine().template retrieveFromHandle<std::vector<lbcrypto::Plaintext>>(h_encoded_data, LogRegBenchmark::tagEncodedResult);
            std::vector<double> decoded = decodeResult(encoded_result, min_count);
            decoded.resize(min_count);
            // convert local format to Test Harness format
            for (std::uint64_t result_sample_i = 0; result_sample_i < min_count; ++result_sample_i)
            {
                if (result.p_buffers[result_sample_i].p && result.p_buffers[result_sample_i].size >= sizeof(double))
                {
                    double *p_result_sample = reinterpret_cast<double *>(result.p_buffers[result_sample_i].p);
                    *p_result_sample        = decoded[result_sample_i];
                } // end if
            } // end for
        } // end if
    } // end if
}

hebench::APIBridge::Handle LogRegBenchmark::encrypt(hebench::APIBridge::Handle h_encoded_data)
{
    // can only encrypt encoded params

    const EncodedOpParams &encoded_params =
        this->getEngine().retrieveFromHandle<EncodedOpParams>(h_encoded_data, tagEncodedOpParams);

    EncryptedOpParams encrypted_params = std::make_tuple(
        m_p_context->context()->Encrypt(m_p_context->publicKey(), std::get<LogRegBenchmarkDescription::Index_W>(encoded_params)),
        m_p_context->context()->Encrypt(m_p_context->publicKey(), std::get<LogRegBenchmarkDescription::Index_b>(encoded_params)),
        encryptInputs(std::get<LogRegBenchmarkDescription::Index_X>(encoded_params)));

    // wrap our internal object into a handle to cross the boundary of the API Bridge
    return this->getEngine().template createHandle<decltype(encrypted_params)>(sizeof(encrypted_params), // size (arbitrary for our usage if we need to)
                                                                               tagEncryptedOpParams, // extra tags
                                                                               std::move(encrypted_params)); // constructor parameters
}

hebench::APIBridge::Handle LogRegBenchmark::decrypt(hebench::APIBridge::Handle h_encrypted_data)
{
    // supports only encrypted result
    const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &encrypted_result =
        this->getEngine().template retrieveFromHandle<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>(h_encrypted_data, LogRegBenchmark::tagEncryptedResult);

    std::vector<lbcrypto::Plaintext> plaintext_result(encrypted_result.size());

    for (size_t i = 0; i < encrypted_result.size(); ++i)
    {
        m_p_context->decrypt(encrypted_result[i], plaintext_result[i]);
    } // end for

    // wrap our internal object into a handle to cross the boundary of the API Bridge
    return this->getEngine().template createHandle<decltype(plaintext_result)>(sizeof(plaintext_result),
                                                                               LogRegBenchmark::tagEncodedResult,
                                                                               std::move(plaintext_result));
}

hebench::APIBridge::Handle LogRegBenchmark::load(const hebench::APIBridge::Handle *p_local_data, uint64_t count)
{
    if (count != 1)
        // we do ops in ciphertext only, so we should get 1 pack
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid number of handles. Expected 1."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    // host is same as remote, so, just duplicate handle to let called be able to destroy input handle
    return this->getEngine().duplicateHandle(p_local_data[0], tagEncryptedOpParams);
}

void LogRegBenchmark::store(hebench::APIBridge::Handle h_remote_data,
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
        p_local_data[0] = this->getEngine().duplicateHandle(h_remote_data, LogRegBenchmark::tagEncryptedResult);
    } // end if
}

hebench::APIBridge::Handle LogRegBenchmark::operate(hebench::APIBridge::Handle h_remote_packed,
                                                    const hebench::APIBridge::ParameterIndexer *p_param_indexers)
{
    EncryptedOpParams &remote =
        this->getEngine().retrieveFromHandle<EncryptedOpParams>(h_remote_packed, tagEncryptedOpParams);

    // extract our internal representation from handle
    const std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> &cipher_input_batches =
        std::get<LogRegBenchmarkDescription::Index_X>(remote);
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &cipher_W = std::get<LogRegBenchmarkDescription::Index_W>(remote);
    // make a copy of the bias to be able to operate without modifying the original
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &cipher_b = std::get<LogRegBenchmarkDescription::Index_b>(remote);

    // validate the indexers

    // this method does not support indexing portions of the batch
    if (p_param_indexers[LogRegBenchmarkDescription::Index_X].value_index != 0
        || (this->getDescriptor().category == hebench::APIBridge::Category::Offline
            && p_param_indexers[LogRegBenchmarkDescription::Index_X].batch_size != m_sample_size)
        || (this->getDescriptor().category == hebench::APIBridge::Category::Latency
            && p_param_indexers[LogRegBenchmarkDescription::Index_X].batch_size != 1))
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid indexer range for parameter " + std::to_string(LogRegBenchmarkDescription::Index_X) + " detected."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    const int max_threads           = m_num_threads;
    const int old_dyn_adjustment    = omp_get_dynamic();
    const int old_max_active_levels = omp_get_max_active_levels();
    const int old_nested_value      = omp_get_nested();
    omp_set_dynamic(false);
    omp_set_nested(true);
    omp_set_max_active_levels(2);

    // compute optimal number of threads per nesting level to avoid idle threads
    int th_lvl[2];
    th_lvl[0] = static_cast<int>(cipher_input_batches.size());
    if (th_lvl[0] < 1)
        th_lvl[0] = 1;
    if (th_lvl[0] > max_threads)
        th_lvl[0] = max_threads;
    th_lvl[1] = max_threads / th_lvl[0];
    if (cipher_input_batches.empty() || th_lvl[1] < 1)
        th_lvl[1] = 1;
    if (!cipher_input_batches.empty()
        && th_lvl[1] > static_cast<int>(cipher_input_batches.front().size()))
        th_lvl[1] = static_cast<int>(cipher_input_batches.front().size());

    std::exception_ptr p_ex;
    std::mutex mtx_ex;

    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> retval(cipher_input_batches.size());
#pragma omp parallel for num_threads(th_lvl[0])
    for (std::size_t batch_i = 0; batch_i < cipher_input_batches.size(); ++batch_i)
    {
        try
        {
            if (!p_ex)
                retval[batch_i] = doLogReg(cipher_W, cipher_b, cipher_input_batches[batch_i], m_w_params.n(), th_lvl[1]);
        }
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(mtx_ex);
            if (!p_ex)
                p_ex = std::current_exception();
        }
    } // end for

    // restore omp to previous values
    omp_set_max_active_levels(old_max_active_levels);
    omp_set_nested(old_nested_value);
    omp_set_dynamic(old_dyn_adjustment);

    if (p_ex)
        std::rethrow_exception(p_ex);

    // use smart ptr to be able to copy during store phase
    return this->getEngine().createHandle<decltype(retval)>(sizeof(retval),
                                                            tagEncryptedResult,
                                                            std::move(retval));
}

} // namespace ckks
} // namespace pbe
