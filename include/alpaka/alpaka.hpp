/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/UniqueId.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/api/api.hpp"
#include "alpaka/api/cpu.hpp"
#include "alpaka/api/unifiedCudaHip.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/Utility.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/math.hpp"
#include "alpaka/math/constants.hpp"
#include "alpaka/mem/Iter.hpp"
#include "alpaka/onAcc.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/GlobalMem.hpp"
#include "alpaka/onAcc/atomic.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/onHost/Device.hpp"
#include "alpaka/onHost/Platform.hpp"
#include "alpaka/onHost/Queue.hpp"
#include "alpaka/onHost/mem/View.hpp"
#include "alpaka/tag.hpp"
