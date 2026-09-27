use crate::build_runtime_error;
use futures::executor::block_on;
use runmat_value::{LogicalArray, Tensor, Value};
use std::cell::Cell;
use std::sync::{Mutex, MutexGuard, OnceLock};

#[cfg(test)]
#[path = "test_support/owning_tests.rs"]
mod owning_tests;

mod native_logical;

pub fn with_native_logical_test_provider<F, R>(f: F) -> R
where
    F: FnOnce(&'static dyn runmat_accelerate_api::AccelProvider) -> R,
{
    let _guard = accel_test_lock();
    let provider = Box::leak(Box::new(native_logical::NativeLogicalProvider::default()));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    f(provider)
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
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        if host.data.element_type() == runmat_accelerate_api::NumericElementType::F64 {
            anyhow::bail!("f32 test provider cannot store native double payloads");
        }
        self.inner.upload_numeric_raw(host)
    }

    fn upload_logical_raw(
        &self,
        host: &runmat_accelerate_api::HostLogicalTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
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
            .upload_numeric_raw(&runmat_accelerate_api::HostNumericTensorView {
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
            Tensor::new(dense, shape.clone())
                .map_err(|e| build_runtime_error(format!("gather: {e}")).build())
        }
        other => Err(build_runtime_error(format!("gather: unsupported value {other:?}")).build()),
    }
}
