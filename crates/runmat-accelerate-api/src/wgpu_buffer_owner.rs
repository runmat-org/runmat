//! Existing shared buffer owner plus immutable physical-creation scope.
//! Scope is inline: clones add no allocation and preserve the original stamp.
#[cfg(target_arch = "wasm32")]
use std::rc::{Rc as Shared, Weak};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc as Shared, Weak};

#[derive(Clone, Debug)]
pub struct WgpuBufferOwner {
    inner: Shared<wgpu::Buffer>,
    scope: Option<crate::DeviceAllocationScope>,
    device: Option<crate::WgpuDeviceAccess>,
}
impl WgpuBufferOwner {
    /// Own a buffer without recording its device. A consumer then assumes the buffer belongs
    /// to the device it draws or computes on.
    pub fn new(buffer: wgpu::Buffer) -> Self {
        Self {
            inner: Shared::new(buffer),
            scope: crate::current_device_allocation_scope(),
            device: None,
        }
    }
    /// Own a buffer created on `device`. A consumer on another device (a render surface with
    /// its own device, for example) checks it with [`crate::WgpuDeviceAccess::same_device`]
    /// and copies the contents instead of binding a buffer its device cannot use.
    pub fn on_device(buffer: wgpu::Buffer, device: &crate::WgpuDeviceAccess) -> Self {
        Self {
            device: Some(device.clone()),
            ..Self::new(buffer)
        }
    }
    /// The device that created the buffer, when it was recorded.
    pub fn device(&self) -> Option<&crate::WgpuDeviceAccess> {
        self.device.as_ref()
    }
    pub fn creation_scope(&self) -> Option<crate::DeviceAllocationScope> {
        self.scope
    }
    pub fn validate_creation_scope(&self) -> Result<(), crate::StaleDeviceAllocationScope> {
        crate::validate_device_allocation_scope(self.scope)
    }
    pub fn as_ptr(this: &Self) -> *const wgpu::Buffer {
        Shared::as_ptr(&this.inner)
    }
    pub fn strong_count(this: &Self) -> usize {
        Shared::strong_count(&this.inner)
    }
    pub fn ptr_eq(left: &Self, right: &Self) -> bool {
        Shared::ptr_eq(&left.inner, &right.inner)
    }
    pub fn downgrade(this: &Self) -> Weak<wgpu::Buffer> {
        Shared::downgrade(&this.inner)
    }
}
impl std::ops::Deref for WgpuBufferOwner {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl AsRef<wgpu::Buffer> for WgpuBufferOwner {
    fn as_ref(&self) -> &wgpu::Buffer {
        &self.inner
    }
}
impl std::borrow::Borrow<wgpu::Buffer> for WgpuBufferOwner {
    fn borrow(&self) -> &wgpu::Buffer {
        &self.inner
    }
}
