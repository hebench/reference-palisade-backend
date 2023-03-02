
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cstring>

#include "engine/palisade_engine.h"
#include "engine/palisade_error.h"
#include "engine/palisade_version.h"

// include all benchmarks
#include "benchmarks/bfv/palisade_bfv_dot_product_benchmark.h"
#include "benchmarks/bfv/palisade_bfv_element_wise_benchmark.h"
#include "benchmarks/bfv/palisade_bfv_matmult_cipherbatchaxis_benchmark.h"
#include "benchmarks/bfv/palisade_bfv_matmulteip_benchmark.h"
#include "benchmarks/bfv/palisade_bfv_matmultrow_benchmark.h"
#include "benchmarks/bfv/palisade_bfv_matmultval_benchmark.h"
#include "benchmarks/ckks/palisade_ckks_dot_product_benchmark.h"
#include "benchmarks/ckks/palisade_ckks_element_wise_benchmark.h"
#include "benchmarks/ckks/palisade_ckks_eltwiseadd_pc_benchmark.h"
#include "benchmarks/ckks/palisade_ckks_logreg_benchmark.h"
#include "benchmarks/ckks/palisade_ckks_matmult_cipherbatchaxis_benchmark.h"
#include "benchmarks/ckks/palisade_ckks_matmulteip_benchmark.h"
#include "benchmarks/ckks/palisade_ckks_matmultrow_benchmark.h"
#include "benchmarks/ckks/palisade_ckks_matmultval_benchmark.h"

//-----------------
// Engine creation
//-----------------

namespace hebench {
namespace cpp {

BaseEngine *createEngine(const std::int8_t *p_buffer, std::uint64_t size)
{
    // backend doesn't need extra init data
    (void)p_buffer;
    (void)size;
    if (HEBENCH_API_VERSION_MAJOR != HEBENCH_API_VERSION_NEEDED_MAJOR
        || HEBENCH_API_VERSION_MINOR != HEBENCH_API_VERSION_NEEDED_MINOR
        || HEBENCH_API_VERSION_REVISION < HEBENCH_API_VERSION_NEEDED_REVISION
        //|| std::strcmp(HEBENCH_API_VERSION_BUILD, HEBENCH_API_VERSION_NEEDED_BUILD) != 0
    )
    {
        std::stringstream ss;
        ss << "Critical: Invalid HEBench API version detected. Required: "
           << HEBENCH_API_VERSION_NEEDED_MAJOR << "." << HEBENCH_API_VERSION_NEEDED_MINOR << "." << HEBENCH_API_VERSION_NEEDED_REVISION
           << ", but " << HEBENCH_API_VERSION_MAJOR << "." << HEBENCH_API_VERSION_MINOR << "." << HEBENCH_API_VERSION_REVISION
           << " received.";
        throw hebench::cpp::HEBenchError(HEBERROR_MSG(ss.str()),
                                         HEBENCH_ECODE_CRITICAL_ERROR);
    } // end if

    return PalisadeEngine::create();
}

void destroyEngine(BaseEngine *p)
{
    PalisadeEngine *_p = dynamic_cast<PalisadeEngine *>(p);
    assert(_p);
    PalisadeEngine::destroy(_p);
}

} // namespace cpp
} // namespace hebench

//---------------------
// class PalisadeEngine
//---------------------

PalisadeEngine *PalisadeEngine::create()
{
    PalisadeEngine *p_retval = new PalisadeEngine();
    p_retval->init();
    return p_retval;
}

void PalisadeEngine::destroy(PalisadeEngine *p)
{
    if (p)
        delete p;
}

PalisadeEngine::PalisadeEngine()
{
}

PalisadeEngine::~PalisadeEngine()
{
}

void PalisadeEngine::init()
{
    // add any new error codes

    addErrorCode(HEBPALISADE_ECODE_PALISADE_ERROR, "PALISADE error");

    // add supported schemes

    addSchemeName(HEBENCH_HE_SCHEME_BFV, "BFV");
    addSchemeName(HEBENCH_HE_SCHEME_CKKS, "CKKS");

    // add supported security

    addSecurityName(HEBPALISADE_HE_SECURITY_128, "128 bits");

    // add the all benchmark descriptors

    addBenchmarkDescription(std::make_shared<pbe::bfv::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Latency, hebench::APIBridge::Workload::EltwiseAdd));
    addBenchmarkDescription(std::make_shared<pbe::ckks::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Latency, hebench::APIBridge::Workload::EltwiseAdd));
    addBenchmarkDescription(std::make_shared<pbe::bfv::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Offline, hebench::APIBridge::Workload::EltwiseAdd));
    addBenchmarkDescription(std::make_shared<pbe::ckks::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Offline, hebench::APIBridge::Workload::EltwiseAdd));

    addBenchmarkDescription(std::make_shared<pbe::bfv::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Latency, hebench::APIBridge::Workload::EltwiseMultiply));
    addBenchmarkDescription(std::make_shared<pbe::ckks::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Latency, hebench::APIBridge::Workload::EltwiseMultiply));
    addBenchmarkDescription(std::make_shared<pbe::bfv::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Offline, hebench::APIBridge::Workload::EltwiseMultiply));
    addBenchmarkDescription(std::make_shared<pbe::ckks::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Offline, hebench::APIBridge::Workload::EltwiseMultiply));

    addBenchmarkDescription(std::make_shared<pbe::bfv::DotProductBenchmarkDescription>(hebench::APIBridge::Category::Latency));
    addBenchmarkDescription(std::make_shared<pbe::ckks::DotProductBenchmarkDescription>(hebench::APIBridge::Category::Latency));
    addBenchmarkDescription(std::make_shared<pbe::bfv::DotProductBenchmarkDescription>(hebench::APIBridge::Category::Offline));
    addBenchmarkDescription(std::make_shared<pbe::ckks::DotProductBenchmarkDescription>(hebench::APIBridge::Category::Offline));

    addBenchmarkDescription(std::make_shared<pbe::bfv::MatMultCipherBatchAxisBenchmarkDescription>());
    addBenchmarkDescription(std::make_shared<pbe::ckks::MatMultCipherBatchAxisBenchmarkDescription>());
    addBenchmarkDescription(std::make_shared<pbe::bfv::MatMultValBenchmarkDescription>());
    addBenchmarkDescription(std::make_shared<pbe::ckks::MatMultValBenchmarkDescription>());
    addBenchmarkDescription(std::make_shared<pbe::bfv::MatMultRowBenchmarkDescription>());
    addBenchmarkDescription(std::make_shared<pbe::ckks::MatMultRowBenchmarkDescription>());

    addBenchmarkDescription(std::make_shared<pbe::ckks::LogRegBenchmarkDescription>(hebench::APIBridge::Category::Latency));
    addBenchmarkDescription(std::make_shared<pbe::ckks::LogRegBenchmarkDescription>(hebench::APIBridge::Category::Offline, 0));

    addBenchmarkDescription(std::make_shared<pbe::ckks::EltwiseAddPlainCipherBenchmarkDescription>());
    addBenchmarkDescription(std::make_shared<pbe::bfv::MatMultEIPBenchmarkDescription>());
    addBenchmarkDescription(std::make_shared<pbe::ckks::MatMultEIPBenchmarkDescription>());
}
