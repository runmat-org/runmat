use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, NativeDeviceAccess, NativeDeviceApi, NativeDeviceComponent,
};

use super::MxClassId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuLeaseOrigin {
    Borrowed,
    Owned,
}

#[derive(Debug)]
struct GpuLeaseInner {
    handle: GpuTensorHandle,
    origin: GpuLeaseOrigin,
    published: AtomicBool,
}

impl Drop for GpuLeaseInner {
    fn drop(&mut self) {
        if self.origin == GpuLeaseOrigin::Owned && !self.published.load(Ordering::Acquire) {
            if let Some(provider) = runmat_accelerate_api::provider_for_handle(&self.handle) {
                let _ = provider.free(&self.handle);
            }
            runmat_accelerate_api::clear_handle_metadata(&self.handle);
        }
    }
}

/// Invocation-scoped ownership for an `mxGPUArray` payload.
///
/// Inputs borrow an existing runtime handle. Provider allocations remain owned
/// by the call arena until an output is published back to the runtime.
#[derive(Debug, Clone)]
pub struct MxGpuLease {
    inner: Arc<GpuLeaseInner>,
}

impl PartialEq for MxGpuLease {
    fn eq(&self, other: &Self) -> bool {
        self.inner.handle == other.inner.handle
    }
}

impl MxGpuLease {
    pub fn borrowed(handle: GpuTensorHandle) -> Self {
        Self::new(handle, GpuLeaseOrigin::Borrowed)
    }

    pub fn owned(handle: GpuTensorHandle) -> Self {
        Self::new(handle, GpuLeaseOrigin::Owned)
    }

    fn new(handle: GpuTensorHandle, origin: GpuLeaseOrigin) -> Self {
        Self {
            inner: Arc::new(GpuLeaseInner {
                handle,
                origin,
                published: AtomicBool::new(false),
            }),
        }
    }

    pub fn handle(&self) -> &GpuTensorHandle {
        &self.inner.handle
    }

    pub fn publish(&self, shape: &[usize]) -> GpuTensorHandle {
        self.inner.published.store(true, Ordering::Release);
        let mut handle = self.inner.handle.clone();
        handle.shape = shape.to_vec();
        handle
    }

    pub fn duplicate(&self) -> anyhow::Result<Self> {
        let provider = runmat_accelerate_api::provider_for_handle(self.handle())
            .ok_or_else(|| anyhow::anyhow!("GPU array provider is unavailable"))?;
        let duplicate = provider.copy_native_device_buffer(self.handle())?;
        validate_result_like(&duplicate, self.handle(), self.handle().descriptor.storage)?;
        if duplicate.buffer_id == self.handle().buffer_id {
            anyhow::bail!("provider returned the source allocation for an independent GPU copy");
        }
        Ok(Self::owned(duplicate))
    }

    pub fn copy_component(&self, component: NativeDeviceComponent) -> anyhow::Result<Self> {
        let provider = runmat_accelerate_api::provider_for_handle(self.handle())
            .ok_or_else(|| anyhow::anyhow!("GPU array provider is unavailable"))?;
        let copy = provider.copy_native_device_component(self.handle(), component)?;
        validate_result_like(&copy, self.handle(), Some(GpuTensorStorage::Real))?;
        Ok(Self::owned(copy))
    }

    pub fn combine(real: &Self, imaginary: &Self) -> anyhow::Result<Self> {
        let provider = runmat_accelerate_api::provider_for_handle(real.handle())
            .ok_or_else(|| anyhow::anyhow!("GPU array provider is unavailable"))?;
        if provider.device_id() != imaginary.handle().device_id {
            anyhow::bail!("complex GPU components belong to different providers");
        }
        let combined =
            provider.combine_native_device_components(real.handle(), imaginary.handle())?;
        validate_result_like(
            &combined,
            real.handle(),
            Some(GpuTensorStorage::ComplexInterleaved),
        )?;
        Ok(Self::owned(combined))
    }

    pub fn device_address(&self, access: NativeDeviceAccess) -> anyhow::Result<u64> {
        let provider = runmat_accelerate_api::provider_for_handle(self.handle())
            .ok_or_else(|| anyhow::anyhow!("GPU array provider is unavailable"))?;
        if provider.native_device_api() != Some(NativeDeviceApi::Cuda) {
            anyhow::bail!(
                "GPU array belongs to provider '{}', which does not expose CUDA memory",
                provider.device_info()
            );
        }
        let exported = provider.export_native_device_buffer(self.handle(), access)?;
        runmat_accelerate_api::validate_native_device_buffer(
            self.handle(),
            &exported,
            NativeDeviceApi::Cuda,
        )?;
        Ok(exported.device_address)
    }
}

fn validate_result_like(
    result: &GpuTensorHandle,
    source: &GpuTensorHandle,
    storage: Option<GpuTensorStorage>,
) -> anyhow::Result<()> {
    let element_type = source
        .descriptor
        .element_type
        .ok_or_else(|| anyhow::anyhow!("GPU array has no numeric element type"))?;
    let storage = storage.ok_or_else(|| anyhow::anyhow!("GPU array has no storage layout"))?;
    runmat_accelerate_api::validate_native_device_result(
        result,
        source.device_id,
        &source.shape,
        element_type,
        storage,
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct MxGpuArray {
    pub class_id: MxClassId,
    pub lease: MxGpuLease,
}
