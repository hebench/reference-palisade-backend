#!/bin/bash
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e

sed -i 's/CMAKE_BINARY_DIR/CMAKE_CURRENT_BINARY_DIR/g' $1
sed -i 's/CMAKE_SOURCE_DIR/CMAKE_CURRENT_SOURCE_DIR/g' $1
