/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Acc.hpp"
#include "alpaka/Device.hpp"
#include "alpaka/Platform.hpp"
#include "alpaka/Queue.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/api/api.hpp"
#include "alpaka/api/cpu.hpp"
#include "alpaka/api/cuda.hpp"
#include "alpaka/computeApi.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/hostApi.hpp"
#include "alpaka/mem/Buffer.hpp"
#include "alpaka/mem/Iter.hpp"
