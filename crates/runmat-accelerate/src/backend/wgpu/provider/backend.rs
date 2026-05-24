// Internal note: this file has become a bit too large.
// Subsequent provider call implementations that would otherwise
// be added in this file should, going forwards, be added to
// ./provider/*.rs instead. This module will be refactored into
// submodules in that manner in the future.

use anyhow::{anyhow, ensure, Result};
use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use futures::channel::oneshot;
use log::{debug, error, info, warn};
use once_cell::sync::OnceCell;
#[cfg(not(target_arch = "wasm32"))]
use pollster::block_on;
use rand::seq::SliceRandom;
use runmat_accelerate_api::{
    AccelContextHandle, AccelContextKind, AccelDownloadFuture, AccelProvider, AccelProviderFuture,
    ApiDeviceInfo, CorrcoefNormalization, CorrcoefOptions, CorrcoefRows, CovNormalization, CovRows,
    CovarianceOptions, FindDirection, FspecialRequest, GpuTensorHandle, GpuTensorStorage,
    HostTensorOwned, HostTensorView, ImfilterOptions, ImfilterPadding, IsMemberOptions,
    IsMemberResult, MeshgridAxisView, PagefunOp, PagefunRequest, ProviderBandwidth,
    ProviderCholResult, ProviderCondNorm, ProviderConv1dOptions, ProviderConvMode,
    ProviderConvOrientation, ProviderCummaxResult, ProviderCumminResult, ProviderEigResult,
    ProviderFindResult, ProviderHermitianKind, ProviderIirFilterOptions, ProviderIirFilterResult,
    ProviderInvOptions, ProviderLinsolveOptions, ProviderLinsolveResult, ProviderLuResult,
    ProviderMeshgridResult, ProviderNanMode, ProviderNormOrder, ProviderPinvOptions,
    ProviderPolyderQuotient, ProviderPolyfitResult, ProviderPolyvalOptions, ProviderPrecision,
    ProviderQrOptions, ProviderQrPivot, ProviderQrPowerIterResult, ProviderQrResult,
    ProviderScanDirection, ProviderStdNormalization, ProviderSymmetryKind, ReduceDimResult,
    ReductionFlavor, ReductionTwoPassMode, SetdiffOptions, SetdiffResult, SortComparison,
    SortOrder, SortResult, SortRowsColumnSpec, SpawnHandleConcurrency, UnionOptions, UnionResult,
    UniqueOptions, UniqueResult, WgpuBufferRef, WgpuContextHandle,
};
use runmat_builtins::{Tensor, Value};
use runmat_runtime::builtins::common::shape::normalize_scalar_shape;
use runmat_runtime::builtins::image::filters::fspecial::{
    spec_from_request as runtime_fspecial_spec_from_request, FspecialFilterSpec,
};
use runmat_runtime::builtins::image::filters::imfilter::{
    apply_imfilter_tensor as runtime_apply_imfilter_tensor, build_imfilter_plan,
};

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
use runmat_runtime::builtins::math::poly::polyfit::polyfit_host_real_for_provider;
use runmat_runtime::builtins::math::reduction::{compute_median_inplace, matlab_gradient_shape};
use runmat_runtime::RuntimeError;
use runmat_time::Instant;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use tracing::info_span;
use wgpu::util::DeviceExt;

#[path = "ops/constructors.rs"]
mod constructors;
#[path = "ops/context.rs"]
mod context;
#[path = "core.rs"]
mod core;
#[path = "ops/elementwise.rs"]
mod elementwise;
#[path = "ops/fft.rs"]
mod fft;
#[path = "helpers.rs"]
mod helpers;
#[path = "ops/image.rs"]
mod image;
#[path = "ops/indexing.rs"]
mod indexing;
#[path = "init.rs"]
mod init;
#[path = "ops/linalg.rs"]
mod linalg;
#[path = "ops/polynomial.rs"]
mod polynomial;
#[path = "ops/random.rs"]
mod random;
#[path = "ops/reduction.rs"]
mod reduction;
#[path = "ops/signal.rs"]
mod signal;
#[path = "ops/solve.rs"]
mod solve;
#[path = "ops/tensor.rs"]
mod tensor;
#[path = "ops/window.rs"]
mod window;

use self::window::WindowKind;

use crate::backend::wgpu::autotune::AutotuneController;
use crate::backend::wgpu::cache::{
    bind_group::BindGroupCache, key as cache_key, persist as cache_persist,
};
use crate::backend::wgpu::config::{
    self, DEFAULT_REDUCTION_WG, DEFAULT_TWO_PASS_THRESHOLD, MATMUL_TILE, WORKGROUP_SIZE,
};
use crate::backend::wgpu::params::{
    BandwidthParams, Conv1dParams, CummaxParams, CumminParams, CumprodParams, CumsumParams,
    DiffParams, FilterParams, GradientParamsF32, GradientParamsF64, ImageNormalizeUniforms,
    LinearGatherParams, LinearScatterParams, QrPowerIterParams, SymmetryParamsF32,
    SymmetryParamsF64, SyrkParams, IMAGE_NORMALIZE_FLAG_BIAS, IMAGE_NORMALIZE_FLAG_GAIN,
    IMAGE_NORMALIZE_FLAG_GAMMA, SYRK_FLAG_ACCUMULATE, SYRK_FLAG_FILL_BOTH,
};
use crate::backend::wgpu::pipelines::{ImageNormalizeBootstrap, WgpuPipelines};
use crate::backend::wgpu::residency::{BufferResidency, BufferUsageClass};
use crate::backend::wgpu::resources::{KernelResourceRegistry, UniformBufferKey};
use crate::backend::wgpu::shaders::image_normalize::{
    IMAGE_NORMALIZE_SHADER_F32, IMAGE_NORMALIZE_SHADER_F64,
};
use crate::backend::wgpu::shaders::logical::{
    ELEM_EQ_SHADER_F32, ELEM_EQ_SHADER_F64, ELEM_GE_SHADER_F32, ELEM_GE_SHADER_F64,
    ELEM_GT_SHADER_F32, ELEM_GT_SHADER_F64, ELEM_LE_SHADER_F32, ELEM_LE_SHADER_F64,
    ELEM_LT_SHADER_F32, ELEM_LT_SHADER_F64, ELEM_NE_SHADER_F32, ELEM_NE_SHADER_F64,
    LOGICAL_AND_SHADER_F32, LOGICAL_AND_SHADER_F64, LOGICAL_ISFINITE_SHADER_F32,
    LOGICAL_ISFINITE_SHADER_F64, LOGICAL_ISINF_SHADER_F32, LOGICAL_ISINF_SHADER_F64,
    LOGICAL_ISNAN_SHADER_F32, LOGICAL_ISNAN_SHADER_F64, LOGICAL_NOT_SHADER_F32,
    LOGICAL_NOT_SHADER_F64, LOGICAL_OR_SHADER_F32, LOGICAL_OR_SHADER_F64, LOGICAL_XOR_SHADER_F32,
    LOGICAL_XOR_SHADER_F64,
};
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
#[path = "trait_impl.rs"]
mod trait_impl;

use backend_shared::*;
use backend_types::*;
#[cfg(not(target_arch = "wasm32"))]
fn install_device_error_handlers(device: &wgpu::Device) {
    device.on_uncaptured_error(Box::new(|error| {
        error!("WGPU uncaptured error: {:?}", error);
    }));
    device.set_device_lost_callback(|reason, message| {
        error!("WGPU device lost: reason={:?}, message={}", reason, message);
    });
}

#[cfg(target_arch = "wasm32")]
fn install_device_error_handlers(device: &wgpu::Device) {
    device.on_uncaptured_error(Box::new(|error| {
        error!("WGPU uncaptured error (wasm): {:?}", error);
    }));
    debug!("wgpu set_device_lost_callback not supported on wasm targets");
}
