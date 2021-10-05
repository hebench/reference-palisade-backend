
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "benchmarks/ckks/palisade_ckks_eltwiseadd_pc_benchmark.h"
#include "engine/palisade_engine.h"
#include "engine/palisade_error.h"

namespace pbe {
namespace ckks {

//-----------------------------------
// class EltwiseAddPlainCipherBenchmarkDescription
//-----------------------------------

EltwiseAddPlainCipherBenchmarkDescription::EltwiseAddPlainCipherBenchmarkDescription()
{
    // initialize the descriptor for this benchmark
    std::memset(&m_descriptor, 0, sizeof(hebench::APIBridge::BenchmarkDescriptor));
    m_descriptor.workload                         = hebench::APIBridge::Workload::EltwiseAdd;
    m_descriptor.data_type                        = hebench::APIBridge::DataType::Float64;
    m_descriptor.category                         = hebench::APIBridge::Category::Offline;
    m_descriptor.cat_params.offline.data_count[0] = 2;
    m_descriptor.cat_params.offline.data_count[1] = 5;
    m_descriptor.cipher_param_mask                = 1 << 1;
    //
    m_descriptor.scheme   = HEBENCH_HE_SCHEME_CKKS;
    m_descriptor.security = HEBPALISADE_HE_SECURITY_128;
    m_descriptor.other    = 0; // no extras needed for our purpose:
        // Other back-ends can use this field to differentiate between
        // benchmarks for which internal parameters, not specified by
        // other fields of this structure, differ.

    // specify default arguments for this workload:
    hebench::cpp::WorkloadParams::EltwiseAdd default_workload_params;
    default_workload_params.n = 1000;
    default_workload_params.add<std::uint64_t>(EltwiseAddPlainCipherBenchmarkDescription::DefaultPolyModulusDegree, "PolyModulusDegree");
    default_workload_params.add<std::uint64_t>(EltwiseAddPlainCipherBenchmarkDescription::DefaultNumCoefficientModuli, "MultiplicativeDepth");
    default_workload_params.add<std::uint64_t>(EltwiseAddPlainCipherBenchmarkDescription::DefaultScaleExponent, "ScaleBits");
    // total: 4 workload params
    this->addDefaultParameters(default_workload_params);
}

EltwiseAddPlainCipherBenchmarkDescription::~EltwiseAddPlainCipherBenchmarkDescription()
{
    // nothing needed in this example
}

std::string EltwiseAddPlainCipherBenchmarkDescription::getBenchmarkDescription(const hebench::APIBridge::WorkloadParams *p_w_params) const
{
    assert(p_w_params->count >= EltwiseAddPlainCipherBenchmarkDescription::NumWorkloadParams);

    std::size_t pmd        = p_w_params->params[1].u_param;
    std::size_t mult_depth = p_w_params->params[2].u_param;
    std::size_t scale_bits = p_w_params->params[3].u_param;

    std::stringstream ss;
    ss << ", Encryption parameters" << std::endl
       << ", , HE Library, PALISADE 1.11.3" << std::endl
       << ", , Key-switching technique, PALISADE Hybrid" << std::endl
       << ", , Poly modulus degree, " << pmd << std::endl
       << ", , Multiplicative Depth, " << mult_depth << std::endl
       << ", , Scale, 2^" << scale_bits;
    return ss.str();
}

hebench::cpp::BaseBenchmark *EltwiseAddPlainCipherBenchmarkDescription::createBenchmark(hebench::cpp::BaseEngine &engine,
                                                                                        const hebench::APIBridge::WorkloadParams *p_params)
{
    if (!p_params)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid empty workload parameters. This workload requires flexible parameters."),
                                         HEBENCH_ECODE_CRITICAL_ERROR);

    PalisadeEngine &ex_engine = dynamic_cast<PalisadeEngine &>(engine);
    return new EltwiseAddPlainCipherBenchmark(ex_engine, m_descriptor, *p_params);
}

void EltwiseAddPlainCipherBenchmarkDescription::destroyBenchmark(hebench::cpp::BaseBenchmark *p_bench)
{
    EltwiseAddPlainCipherBenchmark *p = dynamic_cast<EltwiseAddPlainCipherBenchmark *>(p_bench);
    if (p)
        delete p;
}

//------------------------
// class EltwiseAddPlainCipherBenchmark
//------------------------

EltwiseAddPlainCipherBenchmark::EltwiseAddPlainCipherBenchmark(PalisadeEngine &engine,
                                                               const hebench::APIBridge::BenchmarkDescriptor &bench_desc,
                                                               const hebench::APIBridge::WorkloadParams &bench_params) :
    hebench::cpp::BaseBenchmark(engine, bench_desc, bench_params)
{
    // validate workload parameters

    // number of workload parameters (1 for eltwise add: n; +3 encryption params)
    if (bench_params.count < EltwiseAddPlainCipherBenchmarkDescription::NumWorkloadParams)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid workload parameters. This workload requires " + std::to_string(EltwiseAddPlainCipherBenchmarkDescription::NumWorkloadParams) + " parameters."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    std::size_t pmd        = bench_params.params[1].u_param;
    std::size_t mult_depth = bench_params.params[2].u_param;
    std::size_t scale_bits = bench_params.params[3].u_param;

    // check values of the workload parameters and make sure they are supported by benchmark:

    hebench::cpp::WorkloadParams::EltwiseAdd w_params(bench_params);

    if (w_params.n <= 0
        || w_params.n - 1 > pmd / 2)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid workload parameters. This workload only supports vectors up to size " + std::to_string(pmd / 2) + "."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    m_p_context = PalisadeContext::createCKKSContext(pmd, mult_depth, scale_bits, w_params.n);
    //    m_p_context->printContextInfo(std::cout);
    //    std::cout << std::endl;
}

EltwiseAddPlainCipherBenchmark::~EltwiseAddPlainCipherBenchmark()
{
    // nothing needed in this example
}

hebench::APIBridge::Handle EltwiseAddPlainCipherBenchmark::encode(const hebench::APIBridge::PackedData *p_parameters)
{
    // To be compatible with decode and the method signature, this encode bundles
    // multiple data packs inside a single handle. Since in this example we are
    // doing 1 plaintext parameter and 1 ciphertext parameter, each bundle will
    // contain a single data pack.
    assert(p_parameters && p_parameters->pack_count > 0 && p_parameters->p_data_packs);
    //PalisadeEngine &engine = reinterpret_cast<PalisadeEngine &>(this->getEngine());

    // bundle multiple data packs together, even though we will have only 1 per call
    std::vector<InternalParams> params(p_parameters->pack_count);
    for (std::size_t datapack_i = 0; datapack_i < p_parameters->pack_count; ++datapack_i)
    {
        const hebench::APIBridge::DataPack &datapack = p_parameters->p_data_packs[datapack_i];
        assert(datapack.buffer_count > 0 && datapack.p_buffers);

        params[datapack_i].samples.resize(datapack.buffer_count);
        params[datapack_i].param_position = datapack.param_position;
        params[datapack_i].tag            = InternalParams::tagPlaintext;
    }

    for (std::size_t datapack_i = 0; datapack_i < p_parameters->pack_count; ++datapack_i)
    {
        const hebench::APIBridge::DataPack &datapack = p_parameters->p_data_packs[datapack_i];
        for (std::uint64_t sample_i = 0; sample_i < datapack.buffer_count; ++sample_i)
        {
            const hebench::APIBridge::NativeDataBuffer &sample_buffer =
                datapack.p_buffers[sample_i];
            assert(sample_buffer.p && sample_buffer.size / sizeof(double) > 0);
            // read from raw data: the data type is Float64 as specified in this benchmark description
            //std::vector<double> clear_text;
            //double *p_row = reinterpret_cast<double *>(sample_buffer.p);
            std::vector<double> clear_text(reinterpret_cast<double *>(sample_buffer.p), reinterpret_cast<double *>(sample_buffer.p) + sample_buffer.size / sizeof(double));
            lbcrypto::Plaintext encoded =
                m_p_context->context()->MakeCKKSPackedPlaintext(clear_text);
            // store the encoded plaintext in the parameter samples
            params[datapack_i].samples[sample_i] = std::make_shared<lbcrypto::Plaintext>(std::move(encoded));
        }
    }

    // wrap our internal object into a handle to cross the boundary of the API Bridge
    return this->getEngine().template createHandle<decltype(params)>(sizeof(lbcrypto::Plaintext) * params.size(), // size (arbitrary for our usage if we need to)
                                                                     InternalParams::tagPlaintext, // extra tags
                                                                     std::move(params)); // constructor parameters
}

void EltwiseAddPlainCipherBenchmark::decode(hebench::APIBridge::Handle h_encoded_data, hebench::APIBridge::PackedData *p_native)
{
    // This decode is able to decode multiple data packs bundled in a single handle

    assert(p_native && p_native->p_data_packs && p_native->pack_count > 0);

    //PalisadeEngine &engine = reinterpret_cast<PalisadeEngine &>(this->getEngine());
    // This method should handle decoding of data encoded using encode(), due to
    // specification stating that encode() and decode() are inverses; as well as
    // handle data decrypted from operation() results.

    // retrieve our internal format object from the handle
    const std::vector<InternalParams> &encoded =
        this->getEngine().template retrieveFromHandle<std::vector<InternalParams>>(h_encoded_data);

    // according to specification, we must decode as much data as possible, where
    // any excess encoded data that won't fit into the pre-allocated native buffer
    // shall be ignored

    std::uint64_t min_datapack_count = std::min(p_native->pack_count, encoded.size());
    for (std::size_t datapack_i = 0; datapack_i < min_datapack_count; ++datapack_i)
    {
        // decode the next data pack
        hebench::APIBridge::DataPack *p_native_datapack = &p_native->p_data_packs[datapack_i];

        // find the encoded data pack corresponding to the requested clear text data pack
        const InternalParams *p_encoded_datapack = nullptr;
        for (std::size_t encoded_i = 0; !p_encoded_datapack && encoded_i < encoded.size(); ++encoded_i)
            if (encoded[encoded_i].param_position == p_native_datapack->param_position)
                p_encoded_datapack = &encoded[encoded_i];

        if (!p_encoded_datapack)
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Encoded datapack not found in handle 'h_encoded_data'."),
                                             HEBENCH_ECODE_INVALID_ARGS);
        if ((p_encoded_datapack->tag & InternalParams::tagPlaintext) != InternalParams::tagPlaintext)
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid tag detected for handle 'h_encoded_data'."),
                                             HEBENCH_ECODE_INVALID_ARGS);

        if (p_native_datapack && p_native_datapack->buffer_count > 0)
        {
            std::uint64_t min_sample_count = std::min(p_native_datapack->buffer_count, encoded[datapack_i].samples.size());
            for (std::uint64_t sample_i = 0; sample_i < min_sample_count; ++sample_i)
            {
                // alias the samples
                hebench::APIBridge::NativeDataBuffer &native_sample = p_native_datapack->p_buffers[sample_i];
                lbcrypto::Plaintext &encoded_sample                 = *reinterpret_cast<lbcrypto::Plaintext *>(p_encoded_datapack->samples[sample_i].get());

                // decode as much as possible
                std::vector<double> decoded;
                //engine.context() decode(encoded_sample, decoded);
                decoded = encoded_sample->GetRealPackedValue();
                std::copy_n(decoded.begin(),
                            std::min(decoded.size(), native_sample.size / sizeof(double)),
                            reinterpret_cast<double *>(native_sample.p));
            }
        }
    }
}

hebench::APIBridge::Handle EltwiseAddPlainCipherBenchmark::encrypt(hebench::APIBridge::Handle h_encoded_parameters)
{
    //PalisadeEngine &engine = reinterpret_cast<PalisadeEngine &>(this->getEngine());

    if ((h_encoded_parameters.tag & InternalParams::tagPlaintext) != InternalParams::tagPlaintext)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid tag detected for handle 'h_encoded_parameters'."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    const std::vector<InternalParams> &encoded_parameters =
        this->getEngine().template retrieveFromHandle<std::vector<InternalParams>>(h_encoded_parameters);

    std::vector<InternalParams> encrypted_parameters(encoded_parameters.size());

    for (std::size_t datapack_i = 0; datapack_i < encoded_parameters.size(); ++datapack_i)
    {
        if ((encoded_parameters[datapack_i].tag & InternalParams::tagPlaintext) != InternalParams::tagPlaintext)
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid tag detected in data pack."),
                                             HEBENCH_ECODE_INVALID_ARGS);

        // encrypt samples into our internal representation

        encrypted_parameters[datapack_i].param_position = encoded_parameters[datapack_i].param_position;
        encrypted_parameters[datapack_i].tag            = InternalParams::tagCiphertext;
        encrypted_parameters[datapack_i].samples.resize(encoded_parameters[datapack_i].samples.size());
        for (size_t sample_i = 0; sample_i < encoded_parameters[datapack_i].samples.size(); sample_i++)
        {
            const lbcrypto::Plaintext &encoded_sample =
                *reinterpret_cast<const lbcrypto::Plaintext *>(encoded_parameters[datapack_i].samples[sample_i].get());
            lbcrypto::Ciphertext<lbcrypto::DCRTPoly> encrypted_sample;
            encrypted_sample = m_p_context->context()->Encrypt(m_p_context->publicKey(), encoded_sample);
            encrypted_parameters[datapack_i].samples[sample_i] =
                std::make_shared<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(std::move(encrypted_sample));
        }
    }

    // wrap our internal object into a handle to cross the boundary of the API Bridge
    return this->getEngine().template createHandle<decltype(encrypted_parameters)>(encrypted_parameters.size(),
                                                                                   InternalParams::tagCiphertext,
                                                                                   std::move(encrypted_parameters));
}

hebench::APIBridge::Handle EltwiseAddPlainCipherBenchmark::decrypt(hebench::APIBridge::Handle h_encrypted_data)
{
    //PalisadeEngine &engine = reinterpret_cast<PalisadeEngine &>(this->getEngine());

    if ((h_encrypted_data.tag & InternalParams::tagCiphertext) != InternalParams::tagCiphertext)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid tag detected for handle 'h_encrypted_data'."),
                                         HEBENCH_ECODE_INVALID_ARGS);

    const std::vector<InternalParams> &encrypted_data =
        this->getEngine().template retrieveFromHandle<std::vector<InternalParams>>(h_encrypted_data);

    std::vector<InternalParams> plaintext_data(encrypted_data.size());

    for (std::size_t datapack_i = 0; datapack_i < encrypted_data.size(); ++datapack_i)
    {
        if ((encrypted_data[datapack_i].tag & InternalParams::tagCiphertext) != InternalParams::tagCiphertext)
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid tag detected in data pack."),
                                             HEBENCH_ECODE_INVALID_ARGS);

        // decrypt samples into our internal representation

        plaintext_data[datapack_i].param_position = encrypted_data[datapack_i].param_position;
        plaintext_data[datapack_i].tag            = InternalParams::tagPlaintext;
        plaintext_data[datapack_i].samples.resize(encrypted_data[datapack_i].samples.size());
        for (size_t sample_i = 0; sample_i < encrypted_data[datapack_i].samples.size(); sample_i++)
        {
            const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &encrypted_sample =
                *reinterpret_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> *>(encrypted_data[datapack_i].samples[sample_i].get());
            lbcrypto::Plaintext decrypted_sample;
            m_p_context->decrypt(encrypted_sample, decrypted_sample);
            plaintext_data[datapack_i].samples[sample_i] =
                std::make_shared<lbcrypto::Plaintext>(std::move(decrypted_sample));
        }
    }

    // wrap our internal object into a handle to cross the boundary of the API Bridge
    return this->getEngine().template createHandle<decltype(plaintext_data)>(plaintext_data.size(),
                                                                             InternalParams::tagPlaintext,
                                                                             std::move(plaintext_data));
}

hebench::APIBridge::Handle EltwiseAddPlainCipherBenchmark::load(const hebench::APIBridge::Handle *p_local_data, uint64_t count)
{
    if (count != 2)
        // we do ops in plaintext and ciphertext, so we should get 2 packs
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid number of handles. Expected 2..."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    if (!p_local_data)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid null array of handles: \"p_local_data\""),
                                         HEBENCH_ECODE_INVALID_ARGS);

    // allocate data for output
    std::vector<InternalParams> loaded_data;

    // bundle copies of the parameters in a single handle
    for (std::size_t handle_i = 0; handle_i < count; ++handle_i)
    {
        const hebench::APIBridge::Handle &handle = p_local_data[handle_i];
        const std::vector<InternalParams> &params =
            this->getEngine().template retrieveFromHandle<std::vector<InternalParams>>(handle);
        // the copy is shallow, but shared_ptr in InternalParams
        // ensures correct destruction using reference counting
        loaded_data.insert(loaded_data.end(), params.begin(), params.end());
    }

    return this->getEngine().template createHandle<decltype(loaded_data)>(loaded_data.size(),
                                                                          InternalParams::tagCiphertext | InternalParams::tagPlaintext,
                                                                          std::move(loaded_data));
}

void EltwiseAddPlainCipherBenchmark::store(hebench::APIBridge::Handle h_remote_data,
                                           hebench::APIBridge::Handle *p_local_data, std::uint64_t count)
{
    if (count > 0 && !p_local_data)
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid null array of handles: \"p_local_data\""),
                                         HEBENCH_ECODE_INVALID_ARGS);

    if (count > 0)
    {
        std::vector<InternalParams> plain_data;
        std::vector<InternalParams> encrypted_data;
        const std::vector<InternalParams> &remote_data =
            this->getEngine().template retrieveFromHandle<std::vector<InternalParams>>(h_remote_data);

        // since remote and host are the same for this example, we just need to return a copy
        // of the remote as local data.
        for (const auto &internal_params : remote_data)
        {
            if ((internal_params.tag & InternalParams::tagCiphertext) == InternalParams::tagCiphertext)
                encrypted_data.push_back(internal_params);
            else if ((internal_params.tag & InternalParams::tagPlaintext) == InternalParams::tagPlaintext)
                plain_data.push_back(internal_params);
            else
            {
                throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Unknown tag detected in data pack"),
                                                 HEBENCH_ECODE_INVALID_ARGS);
            }
        }
        if (!encrypted_data.empty())
            // store encrypted data in first handle, as per docs
            p_local_data[0] = this->getEngine().template createHandle<decltype(encrypted_data)>(encrypted_data.size(),
                                                                                                InternalParams::tagCiphertext,
                                                                                                std::move(encrypted_data));
        if (!plain_data.empty())
        {
            // store plain data in next available handle as per docs
            hebench::APIBridge::Handle *p_h_plain = nullptr;
            if (encrypted_data.empty())
                p_h_plain = &p_local_data[0];
            else if (count > 1)
                p_h_plain = &p_local_data[1];

            if (p_h_plain)
                *p_h_plain = this->getEngine().template createHandle<decltype(plain_data)>(plain_data.size(),
                                                                                           InternalParams::tagPlaintext,
                                                                                           std::move(plain_data));
        }
    }

    // pad with zeros any remaining local handles as per specifications
    for (std::uint64_t i = 2; i < count; ++i)
        std::memset(p_local_data + i, 0, sizeof(hebench::APIBridge::Handle));
}

hebench::APIBridge::Handle EltwiseAddPlainCipherBenchmark::operate(hebench::APIBridge::Handle h_remote_packed,
                                                                   const hebench::APIBridge::ParameterIndexer *p_param_indexers)
{
    //PalisadeEngine &engine = reinterpret_cast<PalisadeEngine &>(this->getEngine());

    if ((h_remote_packed.tag & (InternalParams::tagCiphertext | InternalParams::tagPlaintext)) != (InternalParams::tagCiphertext | InternalParams::tagPlaintext))
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS("Invalid tag detected for handle 'h_remote_packed'."),
                                         HEBENCH_ECODE_INVALID_ARGS);
    const std::vector<InternalParams> &loaded_data =
        this->getEngine().template retrieveFromHandle<std::vector<InternalParams>>(h_remote_packed);

    assert(loaded_data.size() == ParametersCount);

    // retrieve the plaintext parameter and the ciphertext parameter
    const std::vector<std::shared_ptr<void>> *p_params[ParametersCount] = { nullptr, nullptr };

    for (std::size_t i = 0; i < loaded_data.size(); ++i)
    {
        if (loaded_data[i].param_position == 0
            && (loaded_data[i].tag & InternalParams::tagPlaintext) == InternalParams::tagPlaintext)
            p_params[0] = &loaded_data[i].samples; // param 0 is plain text
        else if (loaded_data[i].param_position == 1
                 && (loaded_data[i].tag & InternalParams::tagCiphertext) == InternalParams::tagCiphertext)
            p_params[1] = &loaded_data[i].samples; // param 1 is ciphertext
    }

    // validate extracted parameters
    for (std::size_t i = 0; i < ParametersCount; ++i)
    {
        if (!p_params[i])
        {
            std::stringstream ss;
            ss << "Unable to find operation parameter " << i << " loaded.";
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                             HEBENCH_ECODE_INVALID_ARGS);
        }
    }

    std::uint64_t results_count = 1;
    for (std::size_t param_i = 0; param_i < ParametersCount; ++param_i)
    {
        if (p_param_indexers[param_i].value_index >= p_params[param_i]->size())
        {
            std::stringstream ss;
            ss << "Invalid parameter indexer for operation parameter " << param_i << ". Expected index in range [0, "
               << p_params[param_i]->size() << "), but " << p_param_indexers[param_i].value_index << " received.";
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                             HEBENCH_ECODE_INVALID_ARGS);
        }
        else if (p_param_indexers[param_i].value_index + p_param_indexers[param_i].batch_size > p_params[param_i]->size())
        {
            std::stringstream ss;
            ss << "Invalid parameter indexer for operation parameter " << param_i << ". Expected batch size in range [1, "
               << p_params[param_i]->size() - p_param_indexers[param_i].value_index << "], but " << p_param_indexers[param_i].batch_size << " received.";
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                             HEBENCH_ECODE_INVALID_ARGS);
        }
        results_count *= p_param_indexers[param_i].batch_size; // count the number of results expected
    }

    // allocate space for results
    std::vector<InternalParams> results(ResultComponentsCount);
    results.front().samples.resize(results_count); // ResultComponentsCount == 1 for this workload
    results.front().param_position = 0; // result component
    results.front().tag            = InternalParams::tagCiphertext;

    // perform the actual operation
    // keep in mind the result ordering (for offline test, most significant parameter moves faster)
    std::size_t result_i = 0;
    for (std::uint64_t p0_sample_i = p_param_indexers[0].value_index;
         p0_sample_i < p_param_indexers[0].value_index + p_param_indexers[0].batch_size;
         ++p0_sample_i)
    {
        for (std::uint64_t p1_sample_i = p_param_indexers[1].value_index;
             p1_sample_i < p_param_indexers[1].value_index + p_param_indexers[1].batch_size;
             ++p1_sample_i)
        {
            const lbcrypto::Plaintext &p0 =
                *reinterpret_cast<const lbcrypto::Plaintext *>(p_params[0]->at(p0_sample_i).get());
            const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &c1 =
                *reinterpret_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> *>(p_params[1]->at(p1_sample_i).get());
            lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result =
                m_p_context->context()->EvalAdd(c1, p0);

            // store result in our internal representation
            results.front().samples[result_i] = std::make_shared<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(std::move(result));
            ++result_i;
        }
    }

    // send our internal result across the boundary of the API Bridge as a handle
    return this->getEngine().template createHandle<decltype(results)>(sizeof(lbcrypto::Ciphertext<lbcrypto::DCRTPoly>) * results_count,
                                                                      InternalParams::tagCiphertext,
                                                                      std::move(results));
}

} // namespace ckks
} // namespace pbe
