//! RunMat Accelerate: GPU Acceleration Abstraction Layer
//!
//! Goals:
//! - Provide a backend-agnostic API surface that maps RunMat operations to GPU kernels.
//! - Support multiple backends via features (CUDA, ROCm, Metal, Vulkan, OpenCL, wgpu).
//! - Allow zero-copy interop with `runmat-builtins::Matrix` where possible.
//! - Defer actual kernel authoring to backend crates/modules; this crate defines traits and wiring.

use once_cell::sync::Lazy;
use runmat_value::{Tensor, Value};
use std::sync::RwLock;

pub mod backend;
pub mod fusion;
pub mod fusion_exec;
pub mod fusion_residency;
pub mod graph;
mod host_lu;
mod iir_zero_phase;
mod kernel_math;
pub mod native_auto;
pub mod placement;
pub mod precision;
mod reduction_meta;
pub mod simple_provider;
mod sortrows_host;
pub mod telemetry;
mod wgsl;
#[cfg(target_arch = "wasm32")]
mod web_auto_offload_store;
#[cfg(feature = "wgpu")]
use crate::backend::wgpu::provider::WgpuProvider;
pub use fusion::*;
pub use graph::*;
pub use native_auto::{
    apply_auto_offload_calibration_from_file, auto_offload_report, is_sink, prepare_builtin_args,
    promote_binary, promote_binary_in_context, promote_builtin_binary_in_context,
    promote_builtin_unary_in_context, promote_reduction_args, promote_unary,
    refresh_auto_offload_calibration, reset_auto_offload_log, AutoOffloadCalibrationOutcome,
    AutoOffloadCalibrationSummary, AutoOffloadDecisionEntry, AutoOffloadDisposition,
    AutoOffloadReport, BinaryOp, CachedProviderInfo, CalibrationSource, CalibrationUnitPolicy,
    DecisionReason, ReductionOp, ThresholdBase, ThresholdDelta, ThresholdDeltaEntry,
    ThresholdSnapshot, UnaryOp,
};
pub use placement::{report as placement_report, PlacementReport};
pub use reduction_meta::{value_is_all_keyword, ReductionAxes};

/// Read the already-registered WGPU façade for Core's private command proof.
/// This never initializes a device and is absent in non-WGPU builds.
pub fn registered_wgpu_provider_for_profiling(
) -> Option<&'static dyn runmat_accelerate_api::AccelProvider> {
    #[cfg(feature = "wgpu")]
    {
        backend::wgpu::provider::registered_wgpu_provider()
            .map(|provider| provider as &'static dyn runmat_accelerate_api::AccelProvider)
    }
    #[cfg(not(feature = "wgpu"))]
    {
        None
    }
}

#[cfg(any(feature = "wgpu", all(feature = "cuda", not(target_arch = "wasm32"))))]
use runmat_accelerate_api::AccelProvider;
use serde::{Deserialize, Serialize};
#[cfg(feature = "wgpu")]
use wgpu::PowerPreference;

/// Preferred acceleration provider selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AccelerateProviderPreference {
    Auto,
    Wgpu,
    InProcess,
}

impl Default for AccelerateProviderPreference {
    fn default() -> Self {
        Self::Auto
    }
}

/// Power preference used when initializing a WGPU backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AccelPowerPreference {
    Auto,
    HighPerformance,
    LowPower,
}

impl Default for AccelPowerPreference {
    fn default() -> Self {
        Self::Auto
    }
}

/// Logging verbosity for auto-offload promotion decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum AutoOffloadLogLevel {
    Off,
    Info,
    #[default]
    Trace,
}

/// Configuration passed to the native auto-offload planner.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutoOffloadOptions {
    pub enabled: bool,
    pub calibrate: bool,
    #[serde(default)]
    pub log_level: AutoOffloadLogLevel,
}

impl Default for AutoOffloadOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            calibrate: true,
            log_level: AutoOffloadLogLevel::Trace,
        }
    }
}

static AUTO_OFFLOAD_OPTIONS: Lazy<RwLock<AutoOffloadOptions>> =
    Lazy::new(|| RwLock::new(AutoOffloadOptions::default()));

static API_HOOKS: Lazy<()> = Lazy::new(|| {
    runmat_accelerate_api::register_residency_mark(fusion_residency::mark);
    runmat_accelerate_api::register_residency_clear(fusion_residency::clear_identity);
    runmat_accelerate_api::register_sequence_threshold_provider(sequence_threshold_hint_bridge);
    runmat_accelerate_api::register_automatic_offload_enabled_provider(native_auto::auto_enabled);
    runmat_accelerate_api::register_workgroup_size_hint_provider(workgroup_size_hint_bridge);
});

pub fn ensure_residency_hooks() {
    Lazy::force(&API_HOOKS);
}

fn sequence_threshold_hint_bridge() -> Option<usize> {
    native_auto::sequence_threshold_hint()
}

fn workgroup_size_hint_bridge() -> Option<u32> {
    #[cfg(feature = "wgpu")]
    {
        Some(crate::backend::wgpu::config::effective_workgroup_size())
    }
    #[cfg(not(feature = "wgpu"))]
    {
        None
    }
}

pub fn configure_auto_offload(options: AutoOffloadOptions) {
    ensure_residency_hooks();
    if let Ok(mut guard) = AUTO_OFFLOAD_OPTIONS.write() {
        *guard = options;
    }
}

pub(crate) fn auto_offload_options() -> AutoOffloadOptions {
    AUTO_OFFLOAD_OPTIONS
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_default()
}

/// Initialization options for selecting and configuring the acceleration provider.
#[derive(Debug, Clone)]
pub struct AccelerateInitOptions {
    pub enabled: bool,
    pub provider: AccelerateProviderPreference,
    pub allow_inprocess_fallback: bool,
    pub wgpu_power_preference: AccelPowerPreference,
    pub wgpu_force_fallback_adapter: bool,
    pub auto_offload: AutoOffloadOptions,
}

impl Default for AccelerateInitOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: AccelerateProviderPreference::Auto,
            allow_inprocess_fallback: true,
            wgpu_power_preference: AccelPowerPreference::Auto,
            wgpu_force_fallback_adapter: false,
            auto_offload: AutoOffloadOptions::default(),
        }
    }
}

impl AccelerateInitOptions {
    /// Compatibility providers remain available while ordinary automatic
    /// placement respects the master acceleration switch.
    fn effective_auto_offload(&self) -> AutoOffloadOptions {
        let mut options = self.auto_offload.clone();
        options.enabled &= self.enabled;
        options
    }
}

/// Initialize the global acceleration provider using the supplied options.
pub fn initialize_acceleration_provider_with(options: &AccelerateInitOptions) {
    configure_auto_offload(options.effective_auto_offload());

    #[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
    if options.enabled {
        match backend::cuda::register_cuda_provider() {
            Ok(Some(provider)) => {
                log::info!(
                    "RunMat Accelerate: CUDA extension provider {}",
                    provider.device_info()
                );
            }
            Ok(None) => {
                log::debug!("RunMat Accelerate: CUDA driver is unavailable");
            }
            Err(error) => {
                log::warn!("RunMat Accelerate: CUDA extension provider unavailable: {error}");
            }
        }
    }

    if runmat_accelerate_api::provider().is_some() {
        return;
    }

    if !options.enabled {
        if options.allow_inprocess_fallback {
            simple_provider::register_inprocess_provider();
            log::info!(
                "RunMat Accelerate: acceleration disabled; using in-process provider for compatibility"
            );
        } else {
            log::info!("RunMat Accelerate: acceleration disabled; no provider registered");
        }
        return;
    }

    let registered = {
        #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
        {
            let mut reg = false;
            if matches!(
                options.provider,
                AccelerateProviderPreference::Auto | AccelerateProviderPreference::Wgpu
            ) {
                let wgpu_options = backend::wgpu::provider::WgpuProviderOptions {
                    power_preference: match options.wgpu_power_preference {
                        AccelPowerPreference::Auto => PowerPreference::HighPerformance,
                        AccelPowerPreference::HighPerformance => PowerPreference::HighPerformance,
                        AccelPowerPreference::LowPower => PowerPreference::LowPower,
                    },
                    force_fallback_adapter: options.wgpu_force_fallback_adapter,
                    precision: backend::wgpu::provider::WgpuPrecisionPolicy::Automatic,
                };

                match backend::wgpu::provider::register_wgpu_provider(wgpu_options) {
                    Ok(provider) => {
                        reg = true;
                        announce_wgpu_provider(provider);
                    }
                    Err(err) => {
                        log::warn!(
                            "RunMat Accelerate: failed to initialize WGPU provider, falling back: {err}"
                        );
                    }
                }
            }
            reg
        }
        #[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
        {
            if matches!(
                options.provider,
                AccelerateProviderPreference::Auto | AccelerateProviderPreference::Wgpu
            ) {
                log::info!(
                    "RunMat Accelerate: wasm builds require calling initialize_wgpu_provider_async to enable the WGPU backend"
                );
            }
            false
        }
        #[cfg(not(feature = "wgpu"))]
        {
            if matches!(options.provider, AccelerateProviderPreference::Wgpu) {
                log::warn!(
                    "RunMat Accelerate: WGPU provider requested but crate built without 'wgpu' feature"
                );
            }
            false
        }
    };

    if !registered {
        if options.allow_inprocess_fallback
            || matches!(options.provider, AccelerateProviderPreference::InProcess)
        {
            simple_provider::register_inprocess_provider();
            log::info!("RunMat Accelerate: using in-process acceleration provider");
        } else {
            log::warn!("RunMat Accelerate: no acceleration provider registered");
        }
    }
}

/// Replace the global acceleration provider using the supplied options.
///
/// This is intended for long-lived hosts whose project configuration can change between runtime
/// sessions. Clearing the default selection does not invalidate providers that own live handles;
/// the acceleration API retains those device owners until their handles are no longer needed.
pub fn reinitialize_acceleration_provider_with(options: &AccelerateInitOptions) {
    runmat_accelerate_api::clear_provider();
    initialize_acceleration_provider_with(options);
}

#[cfg(feature = "wgpu")]
fn announce_wgpu_provider(provider: &WgpuProvider) {
    let info = provider.device_info_struct();
    let backend = info.backend.as_deref().unwrap_or("unknown");
    log::info!(
        "RunMat Accelerate: using WGPU provider {} (vendor: {}, backend: {})",
        info.name,
        info.vendor,
        backend
    );
    provider.warmup();
    let (hits, misses) = provider.fused_cache_counters();
    log::info!(
        "RunMat Accelerate: fused pipeline cache after warmup - hits: {}, misses: {}",
        hits,
        misses
    );
}

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
pub async fn initialize_wgpu_provider_async(options: &AccelerateInitOptions) -> anyhow::Result<()> {
    configure_auto_offload(options.effective_auto_offload());

    if runmat_accelerate_api::provider().is_some() {
        return Ok(());
    }

    if !options.enabled {
        if options.allow_inprocess_fallback
            || matches!(options.provider, AccelerateProviderPreference::InProcess)
        {
            simple_provider::register_inprocess_provider();
            log::info!(
                "RunMat Accelerate: acceleration disabled; using in-process acceleration provider"
            );
        } else {
            log::info!("RunMat Accelerate: acceleration disabled; no provider registered");
        }
        return Ok(());
    }

    let mut registered = false;
    if matches!(
        options.provider,
        AccelerateProviderPreference::Auto | AccelerateProviderPreference::Wgpu
    ) {
        let wgpu_options = backend::wgpu::provider::WgpuProviderOptions {
            power_preference: match options.wgpu_power_preference {
                AccelPowerPreference::Auto => PowerPreference::HighPerformance,
                AccelPowerPreference::HighPerformance => PowerPreference::HighPerformance,
                AccelPowerPreference::LowPower => PowerPreference::LowPower,
            },
            force_fallback_adapter: options.wgpu_force_fallback_adapter,
            precision: backend::wgpu::provider::WgpuPrecisionPolicy::Automatic,
        };

        match backend::wgpu::provider::register_wgpu_provider_async(wgpu_options).await {
            Ok(provider) => {
                registered = true;
                announce_wgpu_provider(provider);
            }
            Err(err) => {
                log::warn!(
                    "RunMat Accelerate: failed to initialize WGPU provider, falling back: {err}"
                );
            }
        }
    }

    if !registered {
        if options.allow_inprocess_fallback
            || matches!(options.provider, AccelerateProviderPreference::InProcess)
        {
            simple_provider::register_inprocess_provider();
            log::info!("RunMat Accelerate: using in-process acceleration provider");
        } else {
            log::warn!("RunMat Accelerate: no acceleration provider registered");
        }
    }

    Ok(())
}

/// Initialize the acceleration provider using default options.
pub fn initialize_acceleration_provider() {
    initialize_acceleration_provider_with(&AccelerateInitOptions::default());
}

#[cfg(test)]
mod tests {
    #[test]
    fn master_acceleration_switch_masks_only_effective_auto_placement() {
        for master in [false, true] {
            for requested in [false, true] {
                let options = super::AccelerateInitOptions {
                    enabled: master,
                    auto_offload: super::AutoOffloadOptions {
                        enabled: requested,
                        calibrate: true,
                        log_level: super::AutoOffloadLogLevel::Info,
                    },
                    ..Default::default()
                };
                let effective = options.effective_auto_offload();
                assert_eq!(effective.enabled, master && requested);
                assert!(effective.calibrate);
                assert_eq!(effective.log_level, super::AutoOffloadLogLevel::Info);
                assert_eq!(options.auto_offload.enabled, requested);
                assert!(options.allow_inprocess_fallback);
            }
        }
    }

    #[cfg(feature = "wgpu")]
    use crate::backend::wgpu::cache::key::compute_pipeline_hash_bytes;

    #[test]
    #[cfg(feature = "wgpu")]
    fn elementwise_hash_varies_with_arity() {
        let wg = 256u32;
        let h2 = compute_pipeline_hash_bytes(b"shader", "runmat-fusion-layout-2", Some(wg));
        let h3 = compute_pipeline_hash_bytes(b"shader", "runmat-fusion-layout-3", Some(wg));
        assert_ne!(h2, h3, "hash should differ with input arity");
    }
}

/// Return fused pipeline cache statistics if the active provider exposes them.
#[cfg(feature = "wgpu")]
pub fn provider_cache_stats() -> Option<(u64, u64)> {
    runmat_accelerate_api::provider().map(|p| p.fused_cache_counters())
}

/// High-level device kind. Concrete selection is provided by backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    OpenCl,
    Wgpu,
}

/// Device descriptor used for selection and capabilities query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub kind: DeviceKind,
    pub name: String,
    pub vendor: String,
    pub memory_bytes: Option<u64>,
    pub compute_capability: Option<String>,
}

/// Abstract buffer that may reside on device or be host-pinned.
pub trait BufferHandle: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Abstract matrix allocated on a device backend.
pub trait DeviceMatrix: Send + Sync {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn as_buffer(&self) -> &dyn BufferHandle;
}

/// Core backend interface that concrete backends must implement.
pub trait AccelerateBackend: Send + Sync {
    fn device_info(&self) -> DeviceInfo;

    // Memory
    fn upload_matrix(&self, host: &Tensor) -> anyhow::Result<Box<dyn DeviceMatrix>>;
    fn download_matrix(&self, dev: &dyn DeviceMatrix) -> anyhow::Result<Tensor>;

    // Elementwise
    fn elem_add(
        &self,
        a: &dyn DeviceMatrix,
        b: &dyn DeviceMatrix,
    ) -> anyhow::Result<Box<dyn DeviceMatrix>>;
    fn elem_sub(
        &self,
        a: &dyn DeviceMatrix,
        b: &dyn DeviceMatrix,
    ) -> anyhow::Result<Box<dyn DeviceMatrix>>;
    fn elem_mul(
        &self,
        a: &dyn DeviceMatrix,
        b: &dyn DeviceMatrix,
    ) -> anyhow::Result<Box<dyn DeviceMatrix>>;
    fn elem_ne(
        &self,
        a: &dyn DeviceMatrix,
        b: &dyn DeviceMatrix,
    ) -> anyhow::Result<Box<dyn DeviceMatrix>>;
    fn elem_eq(
        &self,
        a: &dyn DeviceMatrix,
        b: &dyn DeviceMatrix,
    ) -> anyhow::Result<Box<dyn DeviceMatrix>>;
    fn elem_div(
        &self,
        a: &dyn DeviceMatrix,
        b: &dyn DeviceMatrix,
    ) -> anyhow::Result<Box<dyn DeviceMatrix>>;
    fn elem_pow(
        &self,
        a: &dyn DeviceMatrix,
        b: &dyn DeviceMatrix,
    ) -> anyhow::Result<Box<dyn DeviceMatrix>>;

    // Linear algebra (future): matmul, transpose, BLAS/LAPACK analogs
    fn matmul(
        &self,
        a: &dyn DeviceMatrix,
        b: &dyn DeviceMatrix,
    ) -> anyhow::Result<Box<dyn DeviceMatrix>>;
    fn transpose(&self, a: &dyn DeviceMatrix) -> anyhow::Result<Box<dyn DeviceMatrix>>;
}

/// Planner determines whether to execute on CPU or a selected backend.
/// This will eventually consult sizes, heuristics, and device availability.
#[derive(Default)]
pub struct Planner {
    backend: Option<Box<dyn AccelerateBackend>>,
}

impl Planner {
    pub fn new(backend: Option<Box<dyn AccelerateBackend>>) -> Self {
        Self { backend }
    }

    pub fn device(&self) -> Option<&dyn AccelerateBackend> {
        self.backend.as_deref()
    }

    /// Example decision hook: execute elementwise add on GPU if large enough.
    pub fn choose_elem_add(&self, a: &Tensor, b: &Tensor) -> ExecutionTarget {
        if let Some(bk) = &self.backend {
            if a.len() >= 1 << 16 && a.rows() == b.rows() && a.cols() == b.cols() {
                return ExecutionTarget::Gpu(bk.device_info());
            }
        }
        ExecutionTarget::Cpu
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionTarget {
    Cpu,
    Gpu(DeviceInfo),
}

/// High-level façade for accelerated operations, falling back to `runmat-runtime`.
pub struct Accelerator {
    planner: Planner,
}

impl Accelerator {
    pub fn new(planner: Planner) -> Self {
        Self { planner }
    }

    pub async fn elementwise_add(
        &self,
        a: &Value,
        b: &Value,
    ) -> runmat_accelerate_api::ProviderExecutionResult<runmat_runtime::execution::RootedValue>
    {
        let _inputs = runmat_runtime::execution::RootedValueLease::for_values([a, b].into_iter())
            .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)?;
        match (a, b) {
            (Value::GpuTensor(ga), Value::GpuTensor(gb)) => {
                let ha = self.gather_handle(ga).await?;
                let hb = self.gather_handle(gb).await?;
                self.elementwise_add_resolved(&ha, &hb).await
            }
            (Value::GpuTensor(ga), other) => {
                let ha = self.gather_handle(ga).await?;
                self.elementwise_add_resolved(&ha, other).await
            }
            (other, Value::GpuTensor(gb)) => {
                let hb = self.gather_handle(gb).await?;
                self.elementwise_add_resolved(other, &hb).await
            }
            _ => self.elementwise_add_resolved(a, b).await,
        }
    }

    async fn elementwise_add_resolved(
        &self,
        a: &Value,
        b: &Value,
    ) -> runmat_accelerate_api::ProviderExecutionResult<runmat_runtime::execution::RootedValue>
    {
        match (a, b) {
            (Value::Tensor(ma), Value::Tensor(mb)) => match self.planner.choose_elem_add(ma, mb) {
                ExecutionTarget::Cpu => {
                    runmat_runtime::call_builtin_async("plus", &[a.clone(), b.clone()])
                        .await
                        .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)
                }
                ExecutionTarget::Gpu(_) => {
                    let bk = self
                        .planner
                        .device()
                        .ok_or_else(|| anyhow::anyhow!("no backend"))?;
                    let da = bk.upload_matrix(ma)?;
                    let db = bk.upload_matrix(mb)?;
                    let dc = bk.elem_add(da.as_ref(), db.as_ref())?;
                    let out = bk.download_matrix(dc.as_ref())?;
                    runmat_runtime::execution::RootedValue::new(Value::Tensor(out))
                        .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)
                }
            },
            _ => runmat_runtime::call_builtin_async("plus", &[a.clone(), b.clone()])
                .await
                .map_err(runmat_accelerate_api::ProviderExecutionError::from_source),
        }
    }

    async fn gather_handle(
        &self,
        h: &runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Value> {
        runmat_runtime::gather_if_needed_async(&Value::GpuTensor(h.clone()))
            .await
            .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)
    }
}

#[cfg(test)]
mod native_gather_tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{ComplexStorage, ComplexTensor, IntegerComplexStorage, IntegerStorage};

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn accelerator_host_add_publishes_rooted_scalar_and_tensor_results() {
        let accelerator = Accelerator::new(Planner::new(None));
        let tensor = |data| Value::Tensor(Tensor::new(data, vec![1, 2]).unwrap());
        for (left, right, expected) in [
            (Value::Num(2.0), Value::Num(3.0), Value::Num(5.0)),
            (
                Value::Int(runmat_value::IntValue::U64(u64::MAX)),
                Value::Int(runmat_value::IntValue::U64(1)),
                Value::Int(runmat_value::IntValue::U64(u64::MAX)),
            ),
            (
                tensor(vec![1.0, 2.0]),
                tensor(vec![3.0, 4.0]),
                tensor(vec![4.0, 6.0]),
            ),
        ] {
            let result = block_on(accelerator.elementwise_add(&left, &right)).expect("host add");
            assert_eq!(result.value(), &expected);
            result.with_owned(|value| assert_eq!(value, expected));
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn accelerator_gather_preserves_native_complex_integer_and_source_lifetime() {
        crate::simple_provider::register_inprocess_provider();
        let provider = runmat_accelerate_api::provider().expect("in-process provider");
        let expected = ComplexTensor::from_complex_storage(
            ComplexStorage::Integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
                    IntegerStorage::U64(vec![3, 4]),
                )
                .unwrap(),
            ),
            vec![1, 2],
        )
        .unwrap();
        let handle = runmat_runtime::builtins::common::gpu_helpers::upload_complex_tensor(
            provider, &expected,
        )
        .expect("native complex integer upload")
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
        let accelerator = Accelerator::new(Planner::new(None));

        let gathered = block_on(accelerator.gather_handle(&handle)).expect("native gather");
        assert_eq!(gathered, Value::ComplexTensor(expected.clone()));
        let gathered_again = block_on(accelerator.gather_handle(&handle)).expect("source survives");
        assert_eq!(gathered_again, Value::ComplexTensor(expected));
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&handle),
            Some(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        );
        handle.release().expect("release handle");

        let unregistered = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let error = block_on(accelerator.gather_handle(&unregistered.handle)).unwrap_err();
        assert!(std::error::Error::source(&error)
            .unwrap()
            .is::<runmat_runtime::RuntimeError>());
        unregistered
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        assert_eq!(unregistered.probe.retirements(), 0);
        drop(unregistered.handle);
        assert_eq!(unregistered.probe.retirements(), 1);
    }
}
