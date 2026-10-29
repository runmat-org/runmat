use anyhow::{anyhow, ensure, Result};
#[path = "allocation_state.rs"]
mod allocation_state;
#[path = "moments_cache.rs"]
mod moments_cache;
#[path = "owner_shared.rs"]
mod owner_shared;
#[path = "pipeline_cache.rs"]
mod pipeline_cache;
use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use log::{debug, error};
use once_cell::sync::OnceCell;
use rand::seq::SliceRandom;
use runmat_accelerate_api::{
    AccelContextHandle, AccelContextKind, AccelDownloadFuture, AccelProvider, AccelProviderFuture,
    ApiDeviceInfo, CorrcoefNormalization, CorrcoefOptions, CorrcoefRows, CovNormalization, CovRows,
    CovarianceOptions, DeepLearningConv1DCapability, DeepLearningConv1DOperands,
    DeepLearningConv1DPaddingValue, DeepLearningConv1DRequest, DeepLearningLayerNormCapability,
    DeepLearningLayerNormRequest, DeepLearningPool1DCapability, DeepLearningPool1DRequest,
    DeepLearningPool1DResultAxis, DeepLearningPool1DRoute, FindDirection, FspecialRequest,
    GpuTensorHandle, GpuTensorStorage, HostTensorOwned, HostTensorView, ImfilterOptions,
    IsMemberOptions, IsMemberResult, MeshgridAxisView, PagefunOp, PagefunRequest,
    ProviderAdamUpdateRequest, ProviderAdamUpdateResult, ProviderBandwidth,
    ProviderBlackScholesPriceRequest, ProviderBlackScholesPriceResult, ProviderCholResult,
    ProviderCondNorm, ProviderConv1dOptions, ProviderConvMode, ProviderConvOrientation,
    ProviderCovarianceToCorrelationResult, ProviderCrossentropyMode, ProviderCrossentropyRequest,
    ProviderCrossentropyResult, ProviderCummaxResult, ProviderCumminResult, ProviderFindResult,
    ProviderHermitianKind, ProviderIirFilterOptions, ProviderIirFilterResult, ProviderInvOptions,
    ProviderLinsolveOptions, ProviderLinsolveResult, ProviderLuResult, ProviderMeshgridResult,
    ProviderNanMode, ProviderNdgridRequest, ProviderNdgridResult, ProviderNormOrder,
    ProviderPinvOptions, ProviderPolyderQuotient, ProviderPolyfitResult, ProviderPolyvalOptions,
    ProviderPrecision, ProviderQrColumnOrder, ProviderQrOptions, ProviderQrOutputs,
    ProviderQrPowerIterResult, ProviderQrResult, ProviderScanDirection, ProviderSpectralFrameMode,
    ProviderSpectralRange, ProviderSpectralRequest, ProviderSpectralResult,
    ProviderStdNormalization, ProviderSymmetryKind, ReduceDimResult, ReductionFlavor,
    ReductionTwoPassMode, SetdiffOptions, SetdiffResult, SortComparison, SortOrder, SortResult,
    SortRowsColumnSpec, SpawnHandleConcurrency, UnionOptions, UnionResult, UniqueOptions,
    UniqueResult, WgpuBufferRef,
};
use runmat_runtime::builtins::common::shape::normalize_scalar_shape;
use runmat_runtime::builtins::image::filters::fspecial::{
    spec_from_request as runtime_fspecial_spec_from_request, FspecialFilterSpec,
};
use runmat_value::{Tensor, Value};

use runmat_runtime::builtins::math::linalg::ops::{
    mldivide_host_real_for_provider, mrdivide_host_real_for_provider,
};
use runmat_runtime::builtins::math::linalg::solve::cond::cond_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::inv::inv_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::linsolve::linsolve_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::norm::norm_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::pinv::pinv_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::rank::rank_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::rcond::rcond_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::structure::bandwidth::ensure_matrix_shape as ensure_bandwidth_shape;
use runmat_runtime::builtins::math::linalg::structure::ishermitian::ishermitian_host_real_data;
use runmat_runtime::builtins::math::linalg::structure::issymmetric::ensure_matrix_shape as ensure_symmetry_shape;
use runmat_runtime::builtins::math::linalg::structure::symrcm::symrcm_host_real_data;
use runmat_runtime::builtins::math::reduction::{compute_median_inplace, matlab_gradient_shape};
use runmat_runtime::RuntimeError;
use runmat_time::Instant;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};
use tracing::info_span;

#[path = "ops/comms.rs"]
mod comms;
#[path = "ops/constructors.rs"]
mod constructors;
#[path = "ops/context.rs"]
mod context;
#[path = "core.rs"]
mod core;
#[path = "ops/deep_learning.rs"]
mod deep_learning;
#[path = "ops/deep_learning/convolution/mod.rs"]
mod deep_learning_convolution;
#[path = "ops/deep_learning/normalization/mod.rs"]
mod deep_learning_normalization;
#[path = "ops/deep_learning/pooling.rs"]
mod deep_learning_pooling;
#[path = "preparation/deep_learning.rs"]
mod deep_learning_preparation;
#[path = "ops/deep_learning/resource_bound.rs"]
mod deep_learning_resource_bound;
#[path = "ops/deep_learning/route.rs"]
mod deep_learning_route;
#[path = "ops/elementwise.rs"]
mod elementwise;
#[path = "ops/fft/mod.rs"]
mod fft;
#[path = "ops/finance.rs"]
mod finance;
#[path = "ops/floating_cast.rs"]
mod floating_cast;
#[path = "helpers.rs"]
mod helpers;
#[path = "ieee_features.rs"]
mod ieee_features;
#[path = "ops/image.rs"]
mod image;
#[path = "ops/indexing.rs"]
mod indexing;
#[path = "init.rs"]
mod init;
#[path = "ops/integer.rs"]
mod integer;
#[path = "ops/integer_float_sum.rs"]
mod integer_float_sum;
#[path = "ops/interpolation.rs"]
mod interpolation;
#[path = "ops/io.rs"]
mod io;
#[path = "ops/linalg/mod.rs"]
mod linalg;
#[cfg(not(target_arch = "wasm32"))]
#[path = "observation_progress.rs"]
mod observation_progress;
#[path = "pipeline_observation.rs"]
mod pipeline_observation;
#[path = "pipeline_recipe.rs"]
mod pipeline_recipe;
#[path = "ops/polynomial.rs"]
mod polynomial;
#[path = "ops/power.rs"]
mod power;
#[path = "preparation.rs"]
mod preparation;
#[path = "ops/random.rs"]
mod random;
#[path = "real_floating.rs"]
mod real_floating;
#[path = "ops/reduction/mod.rs"]
mod reduction;
#[path = "ops/remainder.rs"]
mod remainder;
#[path = "ops/select.rs"]
mod select;
#[path = "ops/signal.rs"]
mod signal;
#[path = "ops/solve.rs"]
mod solve;
#[path = "ops/sparse/mod.rs"]
mod sparse;
#[path = "ops/storage_move.rs"]
mod storage_move;
#[path = "ops/telemetry.rs"]
mod telemetry;
#[path = "ops/tensor.rs"]
mod tensor;
#[path = "validated_fusion.rs"]
mod validated_fusion;
#[path = "ops/view_plan.rs"]
mod view_plan;
#[path = "vm_region_tail.rs"]
mod vm_region_tail;
#[path = "ops/window.rs"]
mod window;

use self::window::WindowKind;

use crate::backend::wgpu::autotune::AutotuneController;
use crate::backend::wgpu::cache::bind_group::BindGroupCache;
use crate::backend::wgpu::params::{
    BandwidthParams, Conv1dParams, CummaxParams, CumminParams, CumprodParams, CumsumParams,
    DiffParams, FilterParams, GradientParamsF32, GradientParamsF64, Interp1ParamsF32,
    Interp1ParamsF64, LinearGatherParams, LinearScatterParams, QrPowerIterParams,
    SymmetryParamsF32, SymmetryParamsF64, SyrkParams, TrapezoidParamsF32, TrapezoidParamsF64,
    SYRK_FLAG_ACCUMULATE, SYRK_FLAG_FILL_BOTH,
};
use crate::backend::wgpu::pipelines::WgpuPipelines;
use crate::backend::wgpu::residency::BufferUsageClass;
use crate::backend::wgpu::resources::{KernelResourceRegistry, UniformBufferKey};
use crate::backend::wgpu::types::NumericPrecision;
const QR_DEVICE_MAX_COLS: usize = 64;
const QR_DEVICE_MAX_ELEMS: usize = 1_000_000;
use crate::fusion::{active_fusion, active_group_plan_clone};
use crate::host_lu::{lu_factor_host, LuHostFactors};
use crate::sortrows_host::{sort_rows_host, SortRowsHostOutputs};
use crate::telemetry::AccelTelemetry;

#[path = "backend_shared.rs"]
pub(crate) mod backend_shared;
#[path = "backend_types.rs"]
pub(crate) mod backend_types;
#[path = "device_control.rs"]
mod device_control;
#[path = "health.rs"]
mod health;
#[path = "trait_impl.rs"]
mod trait_impl;

pub(super) use backend_shared::*;
pub(super) use backend_types::*;
pub(super) use health::WgpuDeviceHealth;
#[cfg(not(target_arch = "wasm32"))]
fn install_device_error_handlers(
    device: &wgpu::Device,
    allocations: &owner_shared::OwnerShared<allocation_state::AllocationState>,
    health: Arc<WgpuDeviceHealth>,
) -> Arc<std::sync::atomic::AtomicBool> {
    device.on_uncaptured_error(Box::new(|error| {
        error!("WGPU uncaptured error: {:?}", error);
        #[cfg(test)]
        eprintln!("WGPU uncaptured error: {error:?}");
    }));
    let observation_stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let weak = Arc::downgrade(&observation_stopped);
    let allocation_owner = owner_shared::OwnerShared::downgrade(allocations);
    device.set_device_lost_callback(move |reason, message| {
        health.mark_lost(health::loss_reason(reason));
        if let Some(owner) = allocation_owner.upgrade() {
            owner.mark_device_lost();
        }
        if let Some(stopped) = weak.upgrade() {
            stopped.store(true, std::sync::atomic::Ordering::Release);
        }
        let _ = runmat_accelerate_api::allocation::catch_allocation_cleanup_unwind(|| {
            error!("WGPU device lost: reason={:?}, message={}", reason, message);
        });
    });
    observation_stopped
}

#[cfg(target_arch = "wasm32")]
fn install_device_error_handlers(device: &wgpu::Device, _health: Arc<WgpuDeviceHealth>) {
    device.on_uncaptured_error(Box::new(|error| {
        error!("WGPU uncaptured error (wasm): {:?}", error);
    }));
    debug!("wgpu set_device_lost_callback not supported on wasm targets");
}
