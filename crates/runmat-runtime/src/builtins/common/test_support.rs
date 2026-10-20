use crate::build_runtime_error;
use futures::executor::block_on;
use runmat_value::{HostNumericBuffer, LogicalArray, NumericStorage, Tensor, Value};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Admit one ordinary source vector into the canonical rooted builtin carrier.
///
/// Direct builtin unit tests are genuine source boundaries: the temporary
/// vector is not a runtime compatibility representation and is admitted once
/// here before the builtin receives it.
#[cfg(test)]
pub(crate) fn builtin_arguments(values: Vec<Value>) -> crate::call::BuiltinArguments {
    crate::call::BuiltinArguments::from_borrowed(&values, None).expect("ordinary test arguments")
}

#[cfg(test)]
pub(crate) fn remove_object_property_unbounded(
    object: &mut runmat_value::ObjectInstance,
    name: &str,
) -> Option<Value> {
    let mut builder = object
        .clone()
        .into_unbounded_builder()
        .expect("ordinary fixture object generation");
    let removed = builder.remove(name);
    *object = builder.finish();
    removed
}

#[cfg(test)]
pub(crate) fn reclassify_object_unbounded(
    object: &runmat_value::ObjectInstance,
    class_name: &str,
) -> runmat_value::ObjectInstance {
    let mut builder = runmat_value::UnboundedObjectBuilder::with_property_capacity(
        class_name,
        object.properties().len(),
    );
    for entry in object.properties().entries() {
        builder.insert_text(entry.key().clone(), entry.value().clone());
    }
    builder.finish()
}

#[cfg(test)]
pub(crate) fn replace_struct_field_unbounded(
    structure: &mut runmat_value::StructValue,
    name: &str,
    value: Value,
) -> Option<Value> {
    let original = std::mem::take(structure);
    let mut previous = None;
    let mut replacement = Some(value);
    let mut fields = original
        .into_fields_unbounded()
        .map(|(field, current)| {
            if field == name {
                previous = Some(current);
                (
                    field,
                    replacement.take().expect("one matching structure field"),
                )
            } else {
                (field, current)
            }
        })
        .collect::<Vec<_>>();
    if let Some(value) = replacement {
        fields.push((name.to_owned(), value));
    }
    *structure = runmat_value::StructValue::from_fields_unbounded(fields);
    previous
}

#[cfg(test)]
pub(crate) fn remove_struct_field_unbounded(
    structure: &mut runmat_value::StructValue,
    name: &str,
) -> Option<Value> {
    let original = std::mem::take(structure);
    let mut removed = None;
    *structure = runmat_value::StructValue::from_fields_unbounded(
        original
            .into_fields_unbounded()
            .filter_map(|(field, value)| {
                if field == name {
                    removed = Some(value);
                    None
                } else {
                    Some((field, value))
                }
            }),
    );
    removed
}

#[cfg(test)]
#[path = "test_support/owning_tests.rs"]
mod owning_tests;

mod native_logical;
pub mod filesystem;

pub fn with_native_logical_test_provider<F, R>(f: F) -> R
where
    F: FnOnce(&'static dyn runmat_accelerate_api::AccelProvider) -> R,
{
    let _guard = accel_test_lock();
    let provider = Box::leak(Box::new(native_logical::NativeLogicalProvider::default()));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    f(provider)
}

/// Cross-backend comparison budget for shader transcendental operations executed in f32.
///
/// WGSL transcendental intrinsics are implemented by the active graphics backend and can
/// differ from the host libm result by more than basic arithmetic. This remains a strict
/// 1e-4 absolute/relative bound while allowing the same conformance tests to run across
/// Metal, Vulkan, and Direct3D implementations.
pub const GPU_F32_TRANSCENDENTAL_TOLERANCE: f64 = 1.0e-4;

/// Cross-backend comparison budget for shader transcendental operations executed in f64.
pub const GPU_F64_TRANSCENDENTAL_TOLERANCE: f64 = 1.0e-12;

/// Return the absolute and relative comparison bounds for the provider's shader precision.
pub fn gpu_transcendental_tolerances(
    precision: runmat_accelerate_api::ProviderPrecision,
) -> (f64, f64) {
    let tolerance = match precision {
        runmat_accelerate_api::ProviderPrecision::F32 => GPU_F32_TRANSCENDENTAL_TOLERANCE,
        runmat_accelerate_api::ProviderPrecision::F64 => GPU_F64_TRANSCENDENTAL_TOLERANCE,
    };
    (tolerance, tolerance)
}

pub mod fs {
    use std::io;
    use std::path::Path;

    pub fn write(path: impl AsRef<Path>, data: impl AsRef<[u8]>) -> io::Result<()> {
        futures::executor::block_on(runmat_filesystem::write_async(path, data))
    }

    pub fn remove_file(path: impl AsRef<Path>) -> io::Result<()> {
        futures::executor::block_on(runmat_filesystem::remove_file_async(path))
    }

    pub fn read(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
        futures::executor::block_on(runmat_filesystem::read_async(path))
    }

    pub fn read_to_string(path: impl AsRef<Path>) -> io::Result<String> {
        futures::executor::block_on(runmat_filesystem::read_to_string_async(path))
    }

    pub fn create_dir(path: impl AsRef<Path>) -> io::Result<()> {
        futures::executor::block_on(runmat_filesystem::create_dir_async(path))
    }

    pub fn create_dir_all(path: impl AsRef<Path>) -> io::Result<()> {
        futures::executor::block_on(runmat_filesystem::create_dir_all_async(path))
    }
}

/// Ensure an in-process acceleration provider is registered for tests,
/// invoking the supplied closure with the provider trait object.
pub(crate) struct GlobalStateTestGuard {
    _guard: Option<MutexGuard<'static, ()>>,
}

thread_local! {
    static GLOBAL_STATE_TEST_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Serialize tests that mutate process-wide runtime state.
///
/// The guard is re-entrant on the current thread because RNG tests can invoke
/// acceleration-provider test helpers. Tests on different threads still
/// serialize on the same mutex.
pub(crate) fn global_state_test_guard() -> GlobalStateTestGuard {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let guard = GLOBAL_STATE_TEST_DEPTH.with(|depth| {
        let current = depth.get();
        let guard = if current == 0 {
            Some(
                LOCK.get_or_init(|| Mutex::new(()))
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()),
            )
        } else {
            None
        };
        depth.set(current + 1);
        guard
    });
    GlobalStateTestGuard { _guard: guard }
}

impl Drop for GlobalStateTestGuard {
    fn drop(&mut self) {
        GLOBAL_STATE_TEST_DEPTH.with(|depth| {
            let current = depth.get();
            debug_assert!(current > 0, "global test-state guard depth underflow");
            depth.set(current.saturating_sub(1));
        });
    }
}

pub struct AccelTestGuard {
    _guard: GlobalStateTestGuard,
}

impl Drop for AccelTestGuard {
    fn drop(&mut self) {
        runmat_accelerate_api::set_thread_provider(None);
        runmat_accelerate_api::clear_provider();
    }
}

pub fn accel_test_lock() -> AccelTestGuard {
    let guard = global_state_test_guard();
    runmat_accelerate_api::set_thread_provider(None);
    runmat_accelerate_api::clear_provider();
    AccelTestGuard { _guard: guard }
}

#[cfg(feature = "wgpu")]
pub fn wgpu_provider_if_available() -> Option<&'static dyn runmat_accelerate_api::AccelProvider> {
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .ok()
    .map(|provider| provider as &'static dyn runmat_accelerate_api::AccelProvider)
}

pub fn with_test_provider<F, R>(f: F) -> R
where
    F: FnOnce(&'static dyn runmat_accelerate_api::AccelProvider) -> R,
{
    let _guard = accel_test_lock();
    for _ in 0..5 {
        runmat_accelerate::simple_provider::register_inprocess_provider();
        runmat_accelerate::simple_provider::reset_inprocess_rng();
        if let Some(provider) = runmat_accelerate_api::provider() {
            let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
            return f(provider);
        }
        std::thread::yield_now();
    }
    panic!("test provider registered");
}

/// Exact native integer transfer is available, but this owner deliberately
/// withholds physical publication evidence. It keeps pre-effect fallback
/// refusal tests meaningful now that InProcessProvider certifies uploads.
pub struct UncertifiedIntegerProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    upload_evidence_queries: AtomicUsize,
}

impl UncertifiedIntegerProvider {
    pub fn upload_evidence_queries(&self) -> usize {
        self.upload_evidence_queries.load(Ordering::Relaxed)
    }
}

impl runmat_accelerate_api::AccelProvider for UncertifiedIntegerProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn device_info(&self) -> String {
        "uncertified native integer publication fixture".into()
    }

    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        self.inner.precision()
    }

    fn spawn_handle_concurrency(&self) -> runmat_accelerate_api::SpawnHandleConcurrency {
        self.inner.spawn_handle_concurrency()
    }

    fn capability_snapshot(&self) -> runmat_accelerate_api::ProviderCapabilitySnapshot {
        self.inner.capability_snapshot()
    }

    fn capability_snapshot_ref(
        &self,
    ) -> Option<&runmat_accelerate_api::ProviderCapabilitySnapshot> {
        self.inner.capability_snapshot_ref()
    }

    fn query_feasibility(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        self.inner.query_feasibility(query)
    }

    fn upload(
        &self,
        host: &runmat_accelerate_api::HostTensorView<'_>,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn download<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }

    fn upload_numeric_raw(
        &self,
        host: &runmat_accelerate_api::HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<runmat_accelerate_api::GpuTensorHandle>
    {
        self.inner.upload_numeric_raw(host)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.inner.download_numeric(handle)
    }

    fn upload_physical_evidence(
        &self,
        _query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> Option<runmat_accelerate_api::ProviderUploadPhysicalEvidence> {
        self.upload_evidence_queries.fetch_add(1, Ordering::Relaxed);
        None
    }

    fn telemetry_snapshot(&self) -> runmat_accelerate_api::ProviderTelemetry {
        self.inner.telemetry_snapshot()
    }

    fn reset_telemetry(&self) {
        self.inner.reset_telemetry();
        self.upload_evidence_queries.store(0, Ordering::Relaxed);
    }
}

pub fn with_uncertified_integer_provider<F, R>(f: F) -> R
where
    F: FnOnce(
        &'static dyn runmat_accelerate_api::AccelProvider,
        &'static UncertifiedIntegerProvider,
    ) -> R,
{
    let _guard = accel_test_lock();
    let fixture: &'static UncertifiedIntegerProvider =
        Box::leak(Box::new(UncertifiedIntegerProvider {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            upload_evidence_queries: AtomicUsize::new(0),
        }));
    let provider: &'static dyn runmat_accelerate_api::AccelProvider = fixture;
    // SAFETY: the leaked fixture has a fresh device identity and lives for
    // the process lifetime; no unrelated device owner is replaced.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    f(provider, fixture)
}

/// In-process owner that declares F32 as its physical storage precision while retaining exact
/// native-integer transfer support for precision-boundary regression tests.
pub struct F32TestProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
}

impl F32TestProvider {
    fn new() -> Self {
        Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        }
    }
}

impl runmat_accelerate_api::AccelProvider for F32TestProvider {
    fn upload_numeric_raw(
        &self,
        host: &runmat_accelerate_api::HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<runmat_accelerate_api::GpuTensorHandle>
    {
        if host.data.element_type() == runmat_accelerate_api::NumericElementType::F64 {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "f32 test provider cannot store native double payloads",
            )
            .into());
        }
        self.inner.upload_numeric_raw(host)
    }

    fn upload_logical_raw(
        &self,
        host: &runmat_accelerate_api::HostLogicalTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<runmat_accelerate_api::GpuTensorHandle>
    {
        self.inner.upload_logical_raw(host)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.inner.download_numeric(handle)
    }

    fn upload(
        &self,
        host: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        let values: Vec<f32> = host.data.iter().map(|value| *value as f32).collect();
        self.inner
            .upload_numeric_host(&runmat_accelerate_api::HostNumericTensorView {
                data: runmat_accelerate_api::HostNumericDataView::F32(&values),
                shape: host.shape,
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
    }

    fn download<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }

    fn upload_integer(
        &self,
        host: &runmat_accelerate_api::HostIntegerTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.inner.upload_integer(host)
    }

    fn download_integer<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelIntegerDownloadFuture<'a> {
        self.inner.download_integer(handle)
    }

    fn device_info(&self) -> String {
        self.inner.device_info()
    }

    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        runmat_accelerate_api::ProviderPrecision::F32
    }

    fn propose_gpu_random(
        &self,
        request: &runmat_accelerate_api::random::GpuRandomRequest<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::random::GpuRandomPhysicalProposal>,
    > {
        if request.precision != self.precision() {
            return Ok(None);
        }
        self.inner.propose_gpu_random(request)
    }

    fn execute_gpu_random(
        &self,
        request: &runmat_accelerate_api::random::GpuRandomRequest<'_>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<runmat_accelerate_api::GpuTensorHandle>
    {
        if request.precision != self.precision() {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "f32 test provider cannot produce native double random values",
            )
            .into());
        }
        self.inner.execute_gpu_random(request, admission)
    }
}

pub fn with_f32_test_provider<F, R>(f: F) -> R
where
    F: FnOnce(&'static dyn runmat_accelerate_api::AccelProvider) -> R,
{
    let _guard = accel_test_lock();
    let provider: &'static dyn runmat_accelerate_api::AccelProvider =
        Box::leak(Box::new(F32TestProvider::new()));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    f(provider)
}

/// Test provider whose native signal hooks deliberately return a handle with
/// incompatible storage. This exercises rejection cleanup before host fallback.
pub struct RejectingNativeResultProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    rejected_owner: &'static CountingHandleOwner,
}

struct CountingHandleOwner {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    probes: Mutex<Vec<runmat_accelerate_api::test_support::RetirementProbe>>,
}

impl RejectingNativeResultProvider {
    fn new(rejected_owner: &'static CountingHandleOwner) -> Self {
        Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            rejected_owner,
        }
    }

    pub fn free_count(&self) -> usize {
        self.rejected_owner
            .probes
            .lock()
            .unwrap()
            .iter()
            .map(|probe| probe.retirements())
            .sum()
    }

    fn rejected_handle(&self) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        let allocation = self.rejected_owner.owner.allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
        )?;
        self.rejected_owner
            .probes
            .lock()
            .unwrap()
            .push(allocation.probe);
        Ok(allocation.handle)
    }
}

impl runmat_accelerate_api::AccelProvider for RejectingNativeResultProvider {
    fn upload(
        &self,
        host: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn download<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }

    fn device_info(&self) -> String {
        self.inner.device_info()
    }

    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        self.inner.precision()
    }

    fn conv1d(
        &self,
        _signal: &runmat_accelerate_api::GpuTensorHandle,
        _kernel: &runmat_accelerate_api::GpuTensorHandle,
        _options: runmat_accelerate_api::ProviderConv1dOptions,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.rejected_handle()
    }

    fn conv2d(
        &self,
        _signal: &runmat_accelerate_api::GpuTensorHandle,
        _kernel: &runmat_accelerate_api::GpuTensorHandle,
        _mode: runmat_accelerate_api::ProviderConvMode,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.rejected_handle()
    }

    fn cross(
        &self,
        _lhs: &runmat_accelerate_api::GpuTensorHandle,
        _rhs: &runmat_accelerate_api::GpuTensorHandle,
        _dim: Option<usize>,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.rejected_handle()
    }
}

pub fn with_rejecting_native_result_provider<F, R>(f: F) -> R
where
    F: FnOnce(&'static RejectingNativeResultProvider) -> R,
{
    let _guard = accel_test_lock();
    let rejected_owner = Box::leak(Box::new(CountingHandleOwner {
        owner: runmat_accelerate_api::test_support::TestAllocationOwner::new(),
        probes: Mutex::new(Vec::new()),
    }));
    let provider = Box::leak(Box::new(RejectingNativeResultProvider::new(rejected_owner)));
    unsafe {
        runmat_accelerate_api::register_provider(provider);
    }
    let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    f(provider)
}

/// Keep the result's managed roots alive until the host tensor is produced.
pub fn gather_rooted(value: crate::execution::RootedValue) -> Result<Tensor, crate::RuntimeError> {
    value.with_owned(gather)
}

/// Gather a value (recursively) so assertions can operate on host tensors.
pub fn gather(value: Value) -> Result<Tensor, crate::RuntimeError> {
    let gathered = block_on(crate::dispatcher::gather_if_needed_async(&value))?;

    match gathered {
        Value::Tensor(t) => Ok(t),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1])
            .map_err(|e| build_runtime_error(format!("gather: {e}")).build()),
        Value::LogicalArray(LogicalArray { data, shape }) => {
            let dense: Vec<f64> = data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            Tensor::from_shared_host_buffer(
                HostNumericBuffer::from_numeric_storage(NumericStorage::F64(dense)),
                shape,
            )
            .map_err(|e| build_runtime_error(format!("gather: {e}")).build())
        }
        other => Err(build_runtime_error(format!("gather: unsupported value {other:?}")).build()),
    }
}

/// Compare floating-point results with independent absolute and relative error bounds.
///
/// GPU transcendental implementations can differ slightly across shader backends. The
/// absolute bound protects comparisons near zero, while the relative bound scales for
/// outputs whose magnitude is greater than one.
pub fn floats_match(
    actual: f64,
    expected: f64,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    if actual == expected {
        return true;
    }
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    let scale = actual.abs().max(expected.abs());
    (actual - expected).abs() <= absolute_tolerance.max(relative_tolerance * scale)
}

#[cfg(test)]
mod float_comparison_tests {
    use super::floats_match;

    #[test]
    fn floating_comparison_uses_absolute_tolerance_near_zero() {
        assert!(floats_match(4.0e-7, 0.0, 1.0e-6, 1.0e-8));
        assert!(!floats_match(2.0e-6, 0.0, 1.0e-6, 1.0e-8));
    }

    #[test]
    fn floating_comparison_scales_relative_tolerance_with_magnitude() {
        assert!(floats_match(100.004, 100.0, 1.0e-6, 5.0e-5));
        assert!(!floats_match(100.006, 100.0, 1.0e-6, 5.0e-5));
    }

    #[test]
    fn floating_comparison_requires_nonfinite_values_to_match_exactly() {
        assert!(floats_match(f64::INFINITY, f64::INFINITY, 0.0, 0.0));
        assert!(!floats_match(f64::INFINITY, f64::MAX, 1.0, 1.0));
        assert!(!floats_match(f64::NAN, f64::NAN, 1.0, 1.0));
    }
}
