
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ostream>
#include <string>

#include <palisade.h>

#include "hebench/api_bridge/cpp/hebench.hpp"

class PalisadeContext
{
public:
    HEBERROR_DECLARE_CLASS_NAME(PalisadeContext)
    PalisadeContext(const PalisadeContext &) = delete;
    PalisadeContext &operator=(const PalisadeContext &) = delete;

public:
    typedef std::shared_ptr<PalisadeContext> Ptr;

    /**
     * @brief CKKS constructor
     * @param poly_modulus_degree
     * @param num_coeff_moduli Min multiplicative depth.
     * @param scale_bits Scale is going to be 2^scale_bits (also used as the min
     * bits for coefficient moduli).
     * @param[in] sec_level Security level to enforce.
     * @param batch_size Number of items to be contained in a ciphertext vector.
     * It must be less than or equal to `poly_modulus_degree / 2`. If 0, the default
     * value `poly_modulus_degree / 2` is used.
     * @param max_depth Controls when auto relinearization happens. Every ciphertext
     * starts at depth 2, and increases by 1 after every multiplication. When depth
     * is greater than max depth, auto relin happens. Default is 2.
     */
    static PalisadeContext::Ptr createCKKSContext(std::size_t poly_modulus_degree,
                                                  std::size_t num_coeff_moduli,
                                                  std::size_t scale_bits,
                                                  lbcrypto::SecurityLevel sec_level,
                                                  std::size_t batch_size = 0,
                                                  std::size_t max_depth  = 2);
    /**
     * @brief CKKS constructor
     * @param poly_modulus_degree
     * @param num_coeff_moduli Min multiplicative depth.
     * @param scale_bits Scale is going to be 2^scale_bits (also used as the min
     * bits for coefficient moduli).
     * @param batch_size Number of items to be contained in a ciphertext vector.
     * It must be less than or equal to `poly_modulus_degree / 2`. If 0, the default
     * value `poly_modulus_degree / 2` is used.
     * @param max_depth Controls when auto relinearization happens. Every ciphertext
     * starts at depth 2, and increases by 1 after every multiplication. When depth
     * is greater than max depth, auto relin happens. Default is 2.
     * @details This method defaults to 128 bits classic security level.
     */
    static PalisadeContext::Ptr createCKKSContext(std::size_t poly_modulus_degree,
                                                  std::size_t num_coeff_moduli,
                                                  std::size_t scale_bits,
                                                  std::size_t batch_size = 0,
                                                  std::size_t max_depth  = 2)
    {
        return createCKKSContext(poly_modulus_degree, num_coeff_moduli, scale_bits,
                                 lbcrypto::HEStd_128_classic, batch_size, max_depth);
    }

    /**
     * @brief BFV constructor
     * @param poly_modulus_degree
     * @param num_coeff_moduli Min multiplicative depth.
     * @param coeff_moduli_bits Min bits for coefficient moduli.
     * @param[in] sec_level Security level to enforce.
     * @param plaintext_modulus Must be a prime number and 65537 (17 bits plaintext modulus).
     * @param max_depth Controls when auto relinearization happens. Every ciphertext
     * starts at depth 2, and increases by 1 after every multiplication. When depth
     * is greater than max depth, auto relin happens. Default is 2.
     * @details
     * @code
     * sigma = 3.2
     * @endcode
     */
    static PalisadeContext::Ptr createBFVContext(std::size_t poly_modulus_degree,
                                                 std::size_t num_coeff_moduli,
                                                 std::size_t coeff_moduli_bits,
                                                 lbcrypto::SecurityLevel sec_level,
                                                 std::size_t plaintext_modulus = 65537,
                                                 std::size_t max_depth         = 2);
    /**
     * @brief BFV constructor
     * @param poly_modulus_degree
     * @param num_coeff_moduli Min multiplicative depth.
     * @param coeff_moduli_bits Min bits for coefficient moduli.
     * @param plaintext_modulus Must be a prime number and 65537 (17 bits plaintext modulus).
     * @param max_depth Controls when auto relinearization happens. Every ciphertext
     * starts at depth 2, and increases by 1 after every multiplication. When depth
     * is greater than max depth, auto relin happens. Default is 2.
     * @details
     * This method defaults to 128 bits classic security level.
     *
     * @code
     * sigma = 3.2
     * @endcode
     */
    static PalisadeContext::Ptr createBFVContext(std::size_t poly_modulus_degree,
                                                 std::size_t num_coeff_moduli,
                                                 std::size_t coeff_moduli_bits,
                                                 std::size_t plaintext_modulus = 65537,
                                                 std::size_t max_depth         = 2)
    {
        return createBFVContext(poly_modulus_degree, num_coeff_moduli, coeff_moduli_bits,
                                lbcrypto::HEStd_128_classic, plaintext_modulus, max_depth);
    }

    void EvalSumKeyGen();
    void EvalMultKeyGen();
    void EvalAtIndexKeyGen(const std::vector<int32_t> &index_list);

    void decrypt(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &cipher, lbcrypto::Plaintext &plain);
    lbcrypto::Plaintext decrypt(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &cipher);

    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &context() { return *m_context; }
    auto publicKey() const { return m_keys->publicKey; }

    std::size_t getSlotCount() const { return m_slot_count; }
    void printContextInfo(std::ostream &os, const std::string &preamble = std::string());

protected:
    PalisadeContext();
    virtual void initCKKS(size_t poly_modulus_degree, std::size_t num_coeff_moduli,
                          std::size_t scale_bits, lbcrypto::SecurityLevel sec_level,
                          std::size_t batch_size, std::size_t max_depth);
    virtual void initBFV(size_t poly_modulus_degree, std::size_t num_coeff_moduli,
                         std::size_t coeff_moduli_bits, lbcrypto::SecurityLevel sec_level,
                         std::size_t plaintext_modulus, std::size_t max_depth);

private:
    std::unique_ptr<lbcrypto::LPKeyPair<lbcrypto::DCRTPoly>> m_keys;
    std::shared_ptr<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>> m_context;
    std::size_t m_slot_count;
};
