
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hebench/api_bridge/cpp/hebench.hpp"
#include <palisade.h>

#define HEBPALISADE_HE_SECURITY_128 1

class PalisadeEngine : public hebench::cpp::BaseEngine
{
public:
    HEBERROR_DECLARE_CLASS_NAME(PalisadeEngine)

public:
    static PalisadeEngine *create();
    static void destroy(PalisadeEngine *p);

    ~PalisadeEngine() override;

protected:
    PalisadeEngine();

    void init() override;
};
