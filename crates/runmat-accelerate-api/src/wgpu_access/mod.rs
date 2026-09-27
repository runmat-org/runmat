//! One synchronous emission boundary per actual device, shared by provider and
//! rendering consumers. It coordinates WGPU's device-global error scope stack;
//! it is neither an allocation owner nor a GPU scheduler.
//!
//! Raw references exposed by a permit are for trusted, audited lexical driver
//! calls. Do not clone/store raw devices or queues, return an unfinished encoder
//! or pass, or use derived resource error emitters (map/unmap/create_view/etc.)
//! outside a later emission. WGPU's clonable types cannot enforce that contract
//! alone. Nested helpers borrow the same permit; reacquiring a shared lock can
//! deadlock behind a queued writer. No guard may span an await or progress wait.
#[cfg(target_arch = "wasm32")]
use std::rc::Rc as Shared;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc as Shared;
use std::{error::Error, fmt, future::Future, pin::Pin};

mod gate;
mod errors;
mod scopes;
pub use errors::{WgpuScopeErrorKind, WgpuScopeErrors};
#[cfg(test)]
mod tests;
use scopes::{ActiveScopes, DetachedScopes, ScopeDriver};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuAccessError {
    Poisoned,
    /// Browser reentrancy is rejected, never blocked on its own event loop.
    Busy,
    WrongDevice,
    Allocation(crate::allocation::AllocationAcquisitionError),
}
impl fmt::Display for WgpuAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Poisoned => "WGPU emission gate is poisoned",
            Self::Busy => "WGPU emission gate is already exclusively borrowed",
            Self::WrongDevice => "WGPU permit belongs to another device authority",
            Self::Allocation(_) => "WGPU exported allocation is unavailable",
        })
    }
}
impl Error for WgpuAccessError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
type PendingError = Pin<Box<dyn Future<Output = Option<wgpu::Error>> + Send + 'static>>;
#[cfg(target_arch = "wasm32")]
type PendingError = Pin<Box<dyn Future<Output = Option<wgpu::Error>> + 'static>>;

struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    gate: gate::Gate,
}
impl ScopeDriver for Inner {
    type Pending = PendingError;
    fn push(&self, filter: wgpu::ErrorFilter) {
        self.device.push_error_scope(filter);
    }
    fn pop(&self) -> PendingError {
        // Both pinned backends invoke pop here, not when this future is polled.
        Box::pin(self.device.pop_error_scope())
    }
}

/// Clone this authority when sharing a device; never construct a second gate
/// around another clone of the same raw WGPU device.
#[derive(Clone)]
pub struct WgpuDeviceAccess {
    inner: Shared<Inner>,
}
fn same_authority<T>(left: &Shared<T>, right: &Shared<T>) -> bool {
    Shared::ptr_eq(left, right)
}
impl fmt::Debug for WgpuDeviceAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuDeviceAccess").finish_non_exhaustive()
    }
}
impl WgpuDeviceAccess {
    /// Teardown after an emission has returned/unwound and all scopes have been
    /// popped. Recovers native gate poison only for restricted resource cleanup;
    /// ordinary emission remains rejected. Never call from an active permit or
    /// an inline driver callback. Browser reentrant teardown is still rejected.
    pub fn with_cleanup<T>(
        &self,
        cleanup: impl for<'p> FnOnce(&WgpuCleanup<'p>) -> T,
    ) -> Result<T, WgpuAccessError> {
        self.inner.gate.cleanup(|| {
            cleanup(&WgpuCleanup {
                _scope: std::marker::PhantomData,
            })
        })
    }

    /// Take the fresh pair returned by request_device. The queue must belong to
    /// this device and no independently gated raw clones may remain in use.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            inner: Shared::new(Inner {
                device,
                queue,
                gate: gate::Gate::default(),
            }),
        }
    }

    pub fn same_device(&self, other: &Self) -> bool {
        same_authority(&self.inner, &other.inner)
    }

    /// A synchronous lexical emission. Return only completed resources, never
    /// an encoder/pass or a future borrowing driver access. Call nested helpers
    /// with this permit instead of calling with_shared again.
    ///
    /// ```compile_fail
    /// use runmat_accelerate_api::WgpuDeviceAccess;
    /// fn escaped_reference(access: &WgpuDeviceAccess) {
    ///     let _device = access.with_shared(|permit| permit.device());
    /// }
    /// ```
    pub fn with_shared<T>(
        &self,
        operation: impl for<'p> FnOnce(&WgpuEmission<'p>) -> T,
    ) -> Result<T, WgpuAccessError> {
        self.inner
            .gate
            .shared(|| operation(&WgpuEmission { access: self }))
    }

    /// Capture validation/OOM (and browser Internal) errors for this exclusive
    /// synchronous transaction. Do fallible semantic preflight before entering;
    /// do not publish created resources into caches/registries in the closure.
    /// They become available only after the detached receipt resolves. Native
    /// fatal submit/poll failures and actual device loss are separate channels.
    pub fn with_scoped<T>(
        &self,
        operation: impl for<'p> FnOnce(&WgpuEmission<'p>) -> T,
    ) -> Result<WgpuScopeReceipt<T>, WgpuAccessError> {
        let detached = self.inner.gate.exclusive(|| {
            let scopes = ActiveScopes::push(self.inner.as_ref());
            let value = operation(&WgpuEmission { access: self });
            scopes.detach(value)
        })?;
        Ok(WgpuScopeReceipt {
            detached,
            access: self.clone(),
        })
    }

    /// Progress is deliberately outside the emission gate: poll can invoke
    /// callbacks which need fresh access. Never call this from an emission.
    /// Callbacks fired inline by submit/map registration must defer GPU cleanup
    /// until that emission returns, rather than recursively acquiring a permit.
    pub fn poll(&self, maintain: wgpu::Maintain) -> wgpu::MaintainResult {
        self.inner.device.poll(maintain)
    }
}

/// Borrowed authority for either kind of synchronous transaction. No public
/// constructor, clone, lock reacquisition or ownership counter exists here.
pub struct WgpuEmission<'a> {
    access: &'a WgpuDeviceAccess,
}

/// Teardown-only authority. No raw device/queue accessor or allocation/encoding
/// operation is available, so poisoned admission cannot become normal work.
pub struct WgpuCleanup<'a> {
    _scope: std::marker::PhantomData<&'a ()>,
}
impl WgpuCleanup<'_> {
    pub fn unmap_buffer(&self, buffer: &wgpu::Buffer) {
        buffer.unmap();
    }
    pub fn release_buffer(&self, buffer: wgpu::Buffer) {
        drop(buffer);
    }
    pub fn release_shared_buffer(&self, buffer: std::sync::Arc<wgpu::Buffer>) {
        drop(buffer);
    }
}
impl WgpuEmission<'_> {
    /// Clone the authority, not a raw device, when handing completed resources
    /// or deferred work to their owner. Do not reacquire it inside this emission.
    pub fn access(&self) -> &WgpuDeviceAccess {
        self.access
    }
    pub fn device(&self) -> &wgpu::Device {
        &self.access.inner.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.access.inner.queue
    }
    pub fn same_device(&self, other: &WgpuDeviceAccess) -> bool {
        self.access.same_device(other)
    }
    pub fn ensure_device(&self, other: &WgpuDeviceAccess) -> Result<(), WgpuAccessError> {
        if self.same_device(other) {
            Ok(())
        } else {
            Err(WgpuAccessError::WrongDevice)
        }
    }

    /// The encoder and its passes cannot escape this callback through the
    /// return value. Drop all passes before it returns; finish occurs guarded.
    pub fn encode(
        &self,
        label: &str,
        commands: impl FnOnce(&mut wgpu::CommandEncoder),
    ) -> wgpu::CommandBuffer {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        commands(&mut encoder);
        encoder.finish()
    }
}

/// Owns unpublished work and already-popped scope futures, never a gate guard.
/// Dropping this receipt/future releases work normally and cannot strand a
/// pushed device scope. It does not cancel already-issued GPU commands.
#[must_use = "resolve the detached receipt before publishing work, or drop it to cancel"]
pub struct WgpuScopeReceipt<T> {
    detached: DetachedScopes<T, PendingError>,
    access: WgpuDeviceAccess,
}
impl<T> WgpuScopeReceipt<T> {
    pub async fn resolve(self) -> Result<T, WgpuScopeErrors> {
        let result = self.detached.resolve().await;
        drop(self.access);
        result.map_err(WgpuScopeErrors::new)
    }
}
