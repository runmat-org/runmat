use super::*;
use crate::AccelProvider;

/// Owns the allocation, not a current-context scope. Native use must retain the
/// invocation's exact context guard and synchronize before that guard leaves.
#[derive(Clone, Debug)]
pub struct OwnedNativeDeviceBuffer {
    handle: GpuTensorHandle,
    address: NativeDeviceBuffer,
    access: NativeDeviceAccess,
}
impl OwnedNativeDeviceBuffer {
    pub fn address_view(&self) -> NativeDeviceBuffer {
        self.address
    }
    pub fn handle(&self) -> &GpuTensorHandle {
        &self.handle
    }
    pub fn access(&self) -> NativeDeviceAccess {
        self.access
    }
}

/// The sole owning export wrapper. Raw backend addresses are trusted backend
/// output, not proved pointers; shared validation checks what is enforceable.
/// Retention starts before the raw hook and survives its returned address.
pub fn export_native_device_buffer(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    access: NativeDeviceAccess,
) -> anyhow::Result<OwnedNativeDeviceBuffer> {
    anyhow::ensure!(
        crate::provider_is_authorized(provider),
        "native export provider is not selected or authorized by this session"
    );
    crate::observe_execution_device_effect(crate::ExecutionDeviceEffect::RawNativeExport);
    if crate::handle_is_transposed(handle) {
        return Err(crate::unsupported_provider_operation(
            "native address export requires contiguous layout; materialize lazy transpose first",
        ));
    }
    if provider.device_id() != handle.device_id() {
        anyhow::bail!("native export provider does not own the allocation");
    }
    let retained = handle.clone();
    retained.allocation_lease().validate_live()?;
    let api = provider
        .native_device_api()
        .ok_or_else(|| anyhow::anyhow!("provider does not expose a native device API"))?;
    let address = provider.export_native_device_buffer_raw(&retained, access)?;
    validate_native_device_buffer(&retained, &address, api)?;
    if address.byte_length as u64 > retained.physical().payload_bytes() {
        anyhow::bail!("native export exceeds the owning allocation payload");
    }
    retained.allocation_lease().validate_live()?;
    Ok(OwnedNativeDeviceBuffer {
        handle: retained,
        address,
        access,
    })
}
