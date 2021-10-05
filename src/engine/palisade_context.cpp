
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <new>

#include "engine/palisade_context.h"
#include "engine/palisade_error.h"

//-----------------------
// class PalisadeContext
//-----------------------

PalisadeContext::Ptr PalisadeContext::createCKKSContext(std::size_t poly_modulus_degree,
                                                        std::size_t num_coeff_moduli,
                                                        std::size_t scale_bits,
                                                        lbcrypto::SecurityLevel sec_level,
                                                        std::size_t batch_size,
                                                        std::size_t max_depth)
{
    PalisadeContext::Ptr retval = PalisadeContext::Ptr(new PalisadeContext());
    retval->initCKKS(poly_modulus_degree, num_coeff_moduli,
                     scale_bits, sec_level, batch_size, max_depth);
    return retval;
}

PalisadeContext::Ptr PalisadeContext::createBFVContext(std::size_t poly_modulus_degree,
                                                       std::size_t num_coeff_moduli,
                                                       std::size_t coeff_moduli_bits,
                                                       lbcrypto::SecurityLevel sec_level,
                                                       std::size_t plaintext_modulus,
                                                       std::size_t max_depth)
{
    PalisadeContext::Ptr retval = PalisadeContext::Ptr(new PalisadeContext());
    retval->initBFV(poly_modulus_degree, num_coeff_moduli, coeff_moduli_bits,
                    sec_level, plaintext_modulus, max_depth);
    return retval;
}

PalisadeContext::PalisadeContext() :
    m_slot_count(0)
{
}

void PalisadeContext::initCKKS(std::size_t poly_modulus_degree, std::size_t num_coeff_moduli,
                               std::size_t scale_bits, lbcrypto::SecurityLevel sec_level,
                               std::size_t batch_size, std::size_t max_depth)
{
    // CKKS

    assert(max_depth > 1);

    try
    {
        uint32_t actual_batch_size = batch_size > 0 ?
                                         batch_size :
                                         poly_modulus_degree / 2;

        // The following call creates a CKKS crypto context based on the
        // arguments defined above.

        lbcrypto::CryptoContext<lbcrypto::DCRTPoly> crypto_context =
            lbcrypto::CryptoContextFactory<lbcrypto::DCRTPoly>::genCryptoContextCKKS(
                num_coeff_moduli, scale_bits, actual_batch_size, sec_level,
                poly_modulus_degree, lbcrypto::APPROXRESCALE, lbcrypto::HYBRID, 0, max_depth);

        crypto_context->Enable(PKESchemeFeature::ENCRYPTION);
        crypto_context->Enable(PKESchemeFeature::SHE);
        crypto_context->Enable(PKESchemeFeature::LEVELEDSHE);
        lbcrypto::LPKeyPair<lbcrypto::DCRTPoly> local_key = crypto_context->KeyGen();
        lbcrypto::LPKeyPair<lbcrypto::DCRTPoly> *p_key    = new lbcrypto::LPKeyPair<lbcrypto::DCRTPoly>(local_key.publicKey, local_key.secretKey);
        local_key                                         = lbcrypto::LPKeyPair<lbcrypto::DCRTPoly>();
        m_keys                                            = std::unique_ptr<lbcrypto::LPKeyPair<lbcrypto::DCRTPoly>>(p_key);

        m_context = std::make_shared<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>>(crypto_context);
        if (!m_context)
            throw std::bad_alloc();
        m_slot_count = poly_modulus_degree / 2;
    }
    catch (lbcrypto::palisade_error &ex)
    {
        throw hebench::cpp::HEBenchError(ex.what(), HEBPALISADE_ECODE_PALISADE_ERROR);
    }
}

void PalisadeContext::initBFV(std::size_t poly_modulus_degree, std::size_t num_coeff_moduli,
                              std::size_t coeff_moduli_bits, lbcrypto::SecurityLevel sec_level,
                              std::size_t plaintext_modulus, std::size_t max_depth)
{
    // BFV

    assert(max_depth > 1);
    assert(plaintext_modulus == 65537);

    try
    {
        static constexpr double sigma = 3.2;

        // Instantiate the crypto context

        lbcrypto::CryptoContext<lbcrypto::DCRTPoly> crypto_context =
            lbcrypto::CryptoContextFactory<lbcrypto::DCRTPoly>::genCryptoContextBFVrns(
                plaintext_modulus, sec_level, sigma, 0, num_coeff_moduli,
                0, OPTIMIZED, max_depth, 0, coeff_moduli_bits, poly_modulus_degree);

        // Enable features that you wish to use

        crypto_context->Enable(ENCRYPTION);
        crypto_context->Enable(SHE);
        lbcrypto::LPKeyPair<lbcrypto::DCRTPoly> local_key = crypto_context->KeyGen();
        lbcrypto::LPKeyPair<lbcrypto::DCRTPoly> *p_key    = new lbcrypto::LPKeyPair<lbcrypto::DCRTPoly>(local_key.publicKey, local_key.secretKey);
        local_key                                         = lbcrypto::LPKeyPair<lbcrypto::DCRTPoly>();
        m_keys                                            = std::unique_ptr<lbcrypto::LPKeyPair<lbcrypto::DCRTPoly>>(p_key);

        m_context    = std::make_shared<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>>(crypto_context);
        m_slot_count = poly_modulus_degree;
    }
    catch (lbcrypto::palisade_error &ex)
    {
        throw hebench::cpp::HEBenchError(ex.what(), HEBPALISADE_ECODE_PALISADE_ERROR);
    }
}

void PalisadeContext::EvalSumKeyGen()
{
    try
    {
        context()->EvalSumKeyGen(m_keys->secretKey);
    }
    catch (lbcrypto::palisade_error &ex)
    {
        throw hebench::cpp::HEBenchError(ex.what(), HEBPALISADE_ECODE_PALISADE_ERROR);
    }
}

void PalisadeContext::EvalMultKeyGen()
{
    try
    {
        context()->EvalMultKeyGen(m_keys->secretKey);
    }
    catch (lbcrypto::palisade_error &ex)
    {
        throw hebench::cpp::HEBenchError(ex.what(), HEBPALISADE_ECODE_PALISADE_ERROR);
    }
}

void PalisadeContext::EvalAtIndexKeyGen(const std::vector<int32_t> &index_list)
{
    try
    {
        context()->EvalAtIndexKeyGen(m_keys->secretKey, index_list);
    }
    catch (lbcrypto::palisade_error &ex)
    {
        throw hebench::cpp::HEBenchError(ex.what(), HEBPALISADE_ECODE_PALISADE_ERROR);
    }
}

void PalisadeContext::decrypt(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &cipher, lbcrypto::Plaintext &plain)
{
    try
    {
        context()->Decrypt(m_keys->secretKey, cipher, &plain);
    }
    catch (lbcrypto::palisade_error &ex)
    {
        throw hebench::cpp::HEBenchError(ex.what(), HEBPALISADE_ECODE_PALISADE_ERROR);
    }
}

lbcrypto::Plaintext PalisadeContext::decrypt(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &cipher)
{
    lbcrypto::Plaintext retval;
    decrypt(cipher, retval);
    return retval;
}

void PalisadeContext::printContextInfo(std::ostream &os, const std::string &preamble)
{
    std::string scheme = context()->getSchemeId();

    int AUXMODSIZE;
    lbcrypto::KeySwitchTechnique ksTech = lbcrypto::KeySwitchTechnique::BV;

    if (NATIVEINT == 128)
        AUXMODSIZE = 119;
    else
        AUXMODSIZE = 60;

    auto cryptoParams = std::shared_ptr<lbcrypto::LPCryptoParametersRLWE<lbcrypto::DCRTPoly>>();

    if (scheme == "Not")
    {
        cryptoParams =
            std::static_pointer_cast<lbcrypto::LPCryptoParametersBFVrns<lbcrypto::DCRTPoly>>(context()->GetCryptoParameters());
        scheme = "BFVrns";
    }
    else if (scheme == "CKKS")
    {
        cryptoParams =
            std::static_pointer_cast<lbcrypto::LPCryptoParametersCKKS<lbcrypto::DCRTPoly>>(context()->GetCryptoParameters());
        ksTech = std::static_pointer_cast<lbcrypto::LPCryptoParametersCKKS<lbcrypto::DCRTPoly>>(cryptoParams)->GetKeySwitchTechnique();
    }
    else if (scheme == "BGVrns")
    {
        cryptoParams =
            std::static_pointer_cast<lbcrypto::LPCryptoParametersBGVrns<lbcrypto::DCRTPoly>>(context()->GetCryptoParameters());
        ksTech = std::static_pointer_cast<lbcrypto::LPCryptoParametersBGVrns<lbcrypto::DCRTPoly>>(cryptoParams)->GetKeySwitchTechnique();
    }
    else
        scheme = "unknown";

    int numLargeDigits      = 0;
    int multiplicativeDepth = context()->GetCryptoParameters()->GetElementParams()->GetParams().size() - 1;

    if (multiplicativeDepth > 3)
        numLargeDigits = 3;
    else if ((multiplicativeDepth >= 1) && (multiplicativeDepth <= 3))
        numLargeDigits = 2;
    else
        numLargeDigits = 1;

    int extraBits = 0;
    if (ksTech == lbcrypto::KeySwitchTechnique::GHS)
        extraBits = std::ceil(static_cast<double>(std::round(std::log2(context()->GetCryptoParameters()->GetElementParams()->GetModulus().ConvertToDouble()))) / AUXMODSIZE) * AUXMODSIZE;
    else if (ksTech == lbcrypto::KeySwitchTechnique::HYBRID)
        extraBits = ceil(ceil(static_cast<double>(std::round(std::log2(context()->GetCryptoParameters()->GetElementParams()->GetModulus().ConvertToDouble()))) / numLargeDigits) / AUXMODSIZE) * AUXMODSIZE;

    os << std::endl
       << preamble << "Scheme, ";
    os << scheme;
    os << std::endl
       << preamble << "Security Level Standard, ";

    if (cryptoParams)
    {
        switch (cryptoParams->GetStdLevel())
        {
        case lbcrypto::SecurityLevel::HEStd_128_classic:
            os << "128 bits";
            break;
        case lbcrypto::SecurityLevel::HEStd_192_classic:
            os << "192 bits";
            break;
        case lbcrypto::SecurityLevel::HEStd_256_classic:
            os << "256 bits";
            break;
        default:
            os << "unknown";
            break;
        }
    }
    else
        os << "unknown";
    os << std::endl
       << preamble << "Poly modulus degree, " << context()->GetCyclotomicOrder() / 2 << std::endl
       << preamble << "Plain modulus, " << context()->GetEncodingParams()->GetPlaintextModulus() << std::endl
       << preamble << ", Bit count, " << std::floor(std::log2(context()->GetEncodingParams()->GetPlaintextModulus())) + 1 << std::endl
       << preamble << "Coefficient Moduli count, " << context()->GetCryptoParameters()->GetElementParams()->GetParams().size() << std::endl
       << preamble << "";
    for (std::uint64_t i = 0; i < context()->GetCryptoParameters()->GetElementParams()->GetParams().size(); ++i)
        os << ", " << context()->GetCryptoParameters()->GetElementParams()->GetParams()[i]->GetModulus();
    os << std::endl
       << preamble << "";
    for (std::uint64_t i = 0; i < context()->GetCryptoParameters()->GetElementParams()->GetParams().size(); ++i)
        os << ", " << std::round(std::log2(context()->GetCryptoParameters()->GetElementParams()->GetParams()[i]->GetModulus().ConvertToDouble()));
    os << std::endl
       << preamble << "Initial bits in coefficient modulus, " << std::round(std::log2(context()->GetCryptoParameters()->GetElementParams()->GetModulus().ConvertToDouble()));
    os << std::endl
       << preamble << "Final (key-switching technique modified) bits in coefficient modulus, " << std::round(std::log2(context()->GetCryptoParameters()->GetElementParams()->GetModulus().ConvertToDouble())) + extraBits << std::endl;
}
