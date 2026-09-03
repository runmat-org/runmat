use crate::build_runtime_error;
use futures::executor::block_on;
use runmat_accelerate_api::AccelProvider as _;
use runmat_value::{LogicalArray, Tensor, Value};
use std::cell::Cell;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex, MutexGuard, OnceLock,
};

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
    fn upload_numeric(
        &self,
        host: &runmat_accelerate_api::HostNumericTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        if host.data.element_type() == runmat_accelerate_api::NumericElementType::F64 {
            anyhow::bail!("f32 test provider cannot store native double payloads");
        }
        self.inner.upload_numeric(host)
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
            .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
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

    fn free(&self, handle: &runmat_accelerate_api::GpuTensorHandle) -> anyhow::Result<()> {
        self.inner.free(handle)
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
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    free_count: AtomicUsize,
}

impl RejectingNativeResultProvider {
    fn new(rejected_owner: &'static CountingHandleOwner) -> Self {
        Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            rejected_owner,
        }
    }

    pub fn free_count(&self) -> usize {
        self.rejected_owner.free_count.load(Ordering::SeqCst)
    }

    fn rejected_handle(&self) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        let mut handle = self
            .rejected_owner
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            })?;
        handle.descriptor.storage =
            Some(runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved);
        Ok(handle)
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

    fn free(&self, handle: &runmat_accelerate_api::GpuTensorHandle) -> anyhow::Result<()> {
        self.inner.free(handle)
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

impl runmat_accelerate_api::AccelProvider for CountingHandleOwner {
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

    fn free(&self, handle: &runmat_accelerate_api::GpuTensorHandle) -> anyhow::Result<()> {
        self.free_count.fetch_add(1, Ordering::SeqCst);
        self.inner.free(handle)
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
}

pub fn with_rejecting_native_result_provider<F, R>(f: F) -> R
where
    F: FnOnce(&'static RejectingNativeResultProvider) -> R,
{
    let _guard = accel_test_lock();
    let rejected_owner = Box::leak(Box::new(CountingHandleOwner {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        free_count: AtomicUsize::new(0),
    }));
    let provider = Box::leak(Box::new(RejectingNativeResultProvider::new(rejected_owner)));
    unsafe {
        runmat_accelerate_api::register_provider(rejected_owner);
        runmat_accelerate_api::register_provider(provider);
    }
    let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    f(provider)
}

/// Gather a value (recursively) so assertions can operate on host tensors.
pub fn gather(value: Value) -> Result<Tensor, crate::RuntimeError> {
    // Ensure the correct provider is active for GPU handles created by the WGPU backend.
    #[cfg(feature = "wgpu")]
    {
        if let Value::GpuTensor(ref h) = value {
            let active_owner = runmat_accelerate_api::provider()
                .is_some_and(|provider| provider.device_id() == h.device_id);
            if h.device_id != 0 && !active_owner {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    let provider = match &value {
        Value::GpuTensor(handle) => runmat_accelerate_api::provider_for_handle(handle)
            .or_else(runmat_accelerate_api::provider),
        _ => runmat_accelerate_api::provider(),
    };

    #[cfg(not(target_arch = "wasm32"))]
    let gathered = {
        let _guard = runmat_accelerate_api::ThreadProviderGuard::set(provider);
        block_on(crate::dispatcher::gather_if_needed_async(&value))?
    };

    #[cfg(target_arch = "wasm32")]
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
            Tensor::new(dense, shape.clone())
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
