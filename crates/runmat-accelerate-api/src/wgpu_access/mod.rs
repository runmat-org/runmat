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
use wgpu::util::DeviceExt;

mod errors;
mod gate;
mod provenance;
mod scopes;
mod numeric_readback;
pub use errors::{WgpuScopeErrorKind, WgpuScopeErrors};
pub use numeric_readback::{
    WgpuNumericReadbackBuffer, WgpuNumericReadbackSource, WgpuNumericReadbackSubmission,
    WgpuNumericReadbackTimestampResources,
};
#[cfg(test)]
mod tests;
use provenance::DeviceEffectProvenance;
use scopes::{ActiveScopes, DetachedScopes, ScopeDriver};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuAccessError {
    Poisoned,
    /// Browser reentrancy is rejected, never blocked on its own event loop.
    Busy,
    WrongDevice,
    ProviderNotSelected,
    Allocation(crate::allocation::AllocationAcquisitionError),
}
impl fmt::Display for WgpuAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Poisoned => "WGPU emission gate is poisoned",
            Self::Busy => "WGPU emission gate is already exclusively borrowed",
            Self::WrongDevice => "WGPU permit belongs to another device authority",
            Self::ProviderNotSelected => "WGPU device is not selected by the active session",
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
    effects: DeviceEffectProvenance,
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

/// Fresh-device authority retained privately by the canonical provider. It
/// cannot be reissued from a previously cloned `WgpuDeviceAccess`.
pub struct CanonicalWgpuEmissionAuthority {
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
                effects: DeviceEffectProvenance::default(),
            }),
        }
    }

    /// Construct from a genuinely fresh device/queue pair and return the one
    /// access-bound authority used by its canonical provider. Ordinary access
    /// clones cannot mint or recover this authority.
    pub fn new_profiled(
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> (Self, CanonicalWgpuEmissionAuthority) {
        let access = Self::new(device, queue);
        let authority = CanonicalWgpuEmissionAuthority {
            inner: access.inner.clone(),
        };
        (access, authority)
    }

    pub fn has_raw_export(&self) -> bool {
        self.inner.effects.has_raw_export()
    }

    /// One-way catastrophic progress fence. Already-admitted synchronous
    /// permits finish before this returns; later emission is denied. This is
    /// not physical device-loss proof and never blocks cleanup or callbacks.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn fence_emission(&self) {
        self.inner.gate.fence_emission();
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn is_emission_fenced(&self) -> bool {
        self.inner.gate.is_fenced()
    }

    /// A mapped initializer or queue write may have enqueued device work
    /// before this command selected its receipt. No empty completion fence
    /// proves that earlier work is inside a later encoder timestamp interval.
    pub fn has_unmeasured_implicit_transfer(&self) -> bool {
        self.inner.effects.has_implicit_transfer()
    }

    /// Copy-only feature metadata; no raw clonable device reference escapes.
    pub fn enabled_features(&self) -> wgpu::Features {
        self.inner.device.features()
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
        self.check_session_selection()?;
        self.inner.gate.shared(|| {
            operation(&WgpuEmission {
                access: self,
                mode: WgpuEmissionMode::Shared,
                canonical: false,
            })
        })
    }

    /// The canonical provider's lexical emission; the same device-local gate
    /// and error scopes apply. A wrong authority cannot enter the closure.
    pub fn with_profiled_shared<T>(
        &self,
        authority: &CanonicalWgpuEmissionAuthority,
        operation: impl for<'p> FnOnce(&WgpuEmission<'p>) -> T,
    ) -> Result<T, WgpuAccessError> {
        self.check_session_selection()?;
        if !same_authority(&self.inner, &authority.inner) {
            return Err(WgpuAccessError::WrongDevice);
        }
        self.inner.gate.shared(|| {
            operation(&WgpuEmission {
                access: self,
                mode: WgpuEmissionMode::Shared,
                canonical: true,
            })
        })
    }

    /// Capture validation, out-of-memory and internal errors for this exclusive
    /// synchronous transaction. Do fallible semantic preflight before entering;
    /// do not publish created resources into caches/registries in the closure.
    /// They become available only after the detached receipt resolves. Native
    /// fatal submit/poll failures and actual device loss are separate channels.
    pub fn with_scoped<T>(
        &self,
        operation: impl for<'p> FnOnce(&WgpuEmission<'p>) -> T,
    ) -> Result<WgpuScopeReceipt<T>, WgpuAccessError> {
        self.check_session_selection()?;
        let detached = self.inner.gate.exclusive(|| {
            let scopes = ActiveScopes::push(self.inner.as_ref());
            let value = operation(&WgpuEmission {
                access: self,
                mode: WgpuEmissionMode::Scoped,
                canonical: false,
            });
            scopes.detach(value)
        })?;
        Ok(WgpuScopeReceipt {
            detached,
            access: self.clone(),
        })
    }

    pub fn with_profiled_scoped<T>(
        &self,
        authority: &CanonicalWgpuEmissionAuthority,
        operation: impl for<'p> FnOnce(&WgpuEmission<'p>) -> T,
    ) -> Result<WgpuScopeReceipt<T>, WgpuAccessError> {
        self.check_session_selection()?;
        if !same_authority(&self.inner, &authority.inner) {
            return Err(WgpuAccessError::WrongDevice);
        }
        let detached = self.inner.gate.exclusive(|| {
            let scopes = ActiveScopes::push(self.inner.as_ref());
            let value = operation(&WgpuEmission {
                access: self,
                mode: WgpuEmissionMode::Scoped,
                canonical: true,
            });
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
    /// Completion registration emits no new device work. It remains available
    /// while ordinary session emission is fenced for reset or cancellation.
    pub fn on_submitted_work_done(
        &self,
        callback: impl FnOnce() + Send + 'static,
    ) -> Result<(), WgpuAccessError> {
        self.inner
            .gate
            .cleanup(|| self.inner.queue.on_submitted_work_done(callback))
    }
    fn check_session_selection(&self) -> Result<(), WgpuAccessError> {
        if crate::device_control::device_access_is_authorized(self) {
            Ok(())
        } else {
            Err(WgpuAccessError::ProviderNotSelected)
        }
    }
}

/// Borrowed authority for either kind of synchronous transaction. No public
/// constructor, clone, lock reacquisition or ownership counter exists here.
pub struct WgpuEmission<'a> {
    access: &'a WgpuDeviceAccess,
    mode: WgpuEmissionMode,
    canonical: bool,
}

/// An immutable fact about the owning access transaction, not cache policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WgpuEmissionMode {
    Shared,
    Scoped,
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
    pub fn destroy_buffer(&self, buffer: &wgpu::Buffer) {
        buffer.destroy();
    }
}
impl WgpuEmission<'_> {
    pub fn mode(&self) -> WgpuEmissionMode {
        self.mode
    }
    /// Clone the authority, not a raw device, when handing completed resources
    /// or deferred work to their owner. Do not reacquire it inside this emission.
    pub fn access(&self) -> &WgpuDeviceAccess {
        self.access
    }
    pub fn device(&self) -> &wgpu::Device {
        if !self.canonical {
            self.access.inner.effects.mark_raw_export();
            crate::observe_execution_device_effect(crate::ExecutionDeviceEffect::RawWgpuExport);
        }
        &self.access.inner.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        self.access.inner.effects.mark_raw_export();
        crate::observe_execution_device_effect(crate::ExecutionDeviceEffect::RawWgpuExport);
        &self.access.inner.queue
    }

    pub fn create_initialized_buffer(
        &self,
        descriptor: &wgpu::util::BufferInitDescriptor<'_>,
    ) -> wgpu::Buffer {
        self.access.inner.effects.mark_implicit_transfer();
        crate::observe_execution_device_effect(crate::ExecutionDeviceEffect::ImplicitQueueTransfer);
        self.access.inner.device.create_buffer_init(descriptor)
    }

    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.access.inner.effects.mark_implicit_transfer();
        crate::observe_execution_device_effect(crate::ExecutionDeviceEffect::ImplicitQueueTransfer);
        self.access.inner.queue.write_buffer(buffer, offset, data);
    }

    /// The one queue-submit path with an access-bound canonical authority.
    /// The Runtime authorizer independently checks the selected Core command
    /// and exact admitted singleton access before any effect is emitted.
    pub fn submit_profiled(
        &self,
        commands: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) -> wgpu::SubmissionIndex {
        if self.canonical {
            crate::observe_canonical_wgpu_submit(self.access);
        } else {
            self.access.inner.effects.mark_raw_export();
            crate::observe_execution_device_effect(crate::ExecutionDeviceEffect::RawWgpuExport);
        }
        self.access.inner.queue.submit(commands)
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
