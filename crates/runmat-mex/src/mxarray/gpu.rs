use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, NativeDeviceAccess, NativeDeviceClass, NativeDeviceComponent,
};

use super::MxClassId;

/// An `mxGPUArray` retains the same allocation authority as runtime values.
/// Publication creates a validated view, not a transfer flag or second lease.
#[derive(Debug, Clone)]
pub struct MxGpuLease {
    handle: GpuTensorHandle,
}

impl PartialEq for MxGpuLease {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl MxGpuLease {
    pub fn borrowed(handle: GpuTensorHandle) -> Self {
        Self { handle }
    }

    pub fn owned(handle: GpuTensorHandle) -> Self {
        Self { handle }
    }

    pub fn handle(&self) -> &GpuTensorHandle {
        &self.handle
    }

    pub fn publish(&self, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        if shape == self.handle.shape() {
            return Ok(self.handle.clone());
        }
        Ok(self
            .handle
            .with_view(self.handle.view().clone().with_shape(shape.to_vec()))?)
    }

    pub fn duplicate(&self) -> anyhow::Result<Self> {
        let provider = runmat_accelerate_api::provider_for_handle(self.handle())
            .ok_or_else(|| anyhow::anyhow!("GPU array provider is unavailable"))?;
        let duplicate = provider.copy_native_device_buffer(self.handle())?;
        validate_result_like(
            &duplicate,
            self.handle(),
            self.handle().descriptor().storage,
        )?;
        if duplicate.shares_allocation_with(self.handle()) {
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
        if provider.device_id() != imaginary.handle().device_id() {
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
        #[cfg(not(target_family = "wasm"))]
        {
            crate::host::native_device_address(self.handle(), access)
        }
        #[cfg(target_family = "wasm")]
        {
            let _ = access;
            anyhow::bail!("native GPU addresses are unavailable in WebAssembly")
        }
    }
}

fn validate_result_like(
    result: &GpuTensorHandle,
    source: &GpuTensorHandle,
    storage: Option<GpuTensorStorage>,
) -> anyhow::Result<()> {
    let class = if runmat_accelerate_api::handle_is_logical(source) {
        NativeDeviceClass::Logical
    } else {
        NativeDeviceClass::Numeric(source.physical().element_type())
    };
    let storage = storage.ok_or_else(|| anyhow::anyhow!("GPU array has no storage layout"))?;
    runmat_accelerate_api::validate_native_device_result(
        result,
        source.device_id(),
        source.shape(),
        class,
        storage,
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct MxGpuArray {
    pub class_id: MxClassId,
    pub lease: MxGpuLease,
}

#[cfg(test)]
mod tests;
