//! Exact, transactional admission for explicit GC roots.

mod allocator;
mod registry;
mod trace_error;
pub use trace_error::RootTraceFailure;
#[cfg(test)]
mod tests;
pub(crate) use registry::RootRegistry;

use allocator::{empty_map, table_layout, RegistryAllocator};
use runmat_execution::resource::ExactShared;
use runmat_value::{ExactHostAllocationClaim, ExactHostAllocationPermit};
use std::alloc::{alloc, dealloc, Layout};
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{GcHandle, Trace, Tracer, GC, ROOT_ADMISSION};
use registry::increment_map;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedRootQuote {
    pub raw_retained_bytes: usize,
    pub retained_claims: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RootAdmissionError {
    CountOverflow,
    LayoutOverflow,
    AllocationDenied,
    AllocationFailed,
    TraceChanged,
    StalePlan,
    InvalidHandle,
    RegistrationCountOverflow,
    RevisionExhausted,
}

impl fmt::Display for RootAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for RootAdmissionError {}

#[derive(Clone, Copy, Debug)]
struct RegistryGrowth {
    capacity: usize,
    layout: Layout,
    retained_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct PreparedRootPlan {
    upper_handles: usize,
    revision: Option<u64>,
    growth: Option<RegistryGrowth>,
    quote: PreparedRootQuote,
}

impl PreparedRootPlan {
    /// Inspect a protected source graph and quote the complete retained root
    /// lease. The source must remain rooted or otherwise collection-safe until
    /// `prepare` finishes; this plan is metadata, not a root.
    pub fn for_traced<T: Trace + ?Sized>(source: &T) -> Result<Self, RootAdmissionError> {
        Self::try_for_traced(|tracer| {
            source.trace(tracer);
            Ok::<_, std::convert::Infallible>(())
        })
        .map_err(trace_error::infallible_source)
    }

    /// Plan using a fallible source traversal. The source owns its scratch
    /// admission; GC counts handles only after that traversal succeeds.
    pub fn try_for_traced<E>(
        mut trace: impl FnMut(&mut dyn Tracer) -> Result<(), E>,
    ) -> Result<Self, RootTraceFailure<E>> {
        let mut counter = CountingTracer {
            count: 0,
            overflow: false,
        };
        trace(&mut counter).map_err(RootTraceFailure::Source)?;
        if counter.overflow {
            return Err(RootAdmissionError::CountOverflow.into());
        }
        if counter.count == 0 {
            return Ok(Self {
                upper_handles: 0,
                revision: None,
                growth: None,
                quote: PreparedRootQuote {
                    raw_retained_bytes: 0,
                    retained_claims: 0,
                },
            });
        }

        let _admission = ROOT_ADMISSION.read();
        let registry = GC.root_ptrs.lock();
        let revision = registry.plan_revision()?;
        let target = registry
            .len()
            .checked_add(counter.count)
            .ok_or(RootAdmissionError::CountOverflow)?;
        let growth = if target > registry.capacity() {
            let layout = table_layout::<GcHandle, usize>(target)
                .map_err(|_| RootAdmissionError::LayoutOverflow)?;
            let retained_bytes = RegistryAllocator::owner_bytes()
                .checked_add(
                    RegistryAllocator::block_bytes(layout)
                        .map_err(|_| RootAdmissionError::LayoutOverflow)?,
                )
                .ok_or(RootAdmissionError::LayoutOverflow)?;
            Some(RegistryGrowth {
                capacity: target,
                layout,
                retained_bytes,
            })
        } else {
            None
        };
        let lease_bytes = ExactShared::<RootLeasePayload>::requested_bytes()
            .checked_add(handle_layout(counter.count)?.size())
            .ok_or(RootAdmissionError::LayoutOverflow)?;
        let raw_retained_bytes = lease_bytes
            .checked_add(growth.map_or(0, |growth| growth.retained_bytes))
            .ok_or(RootAdmissionError::LayoutOverflow)?;
        let retained_claims = 1 + usize::from(growth.is_some());
        Ok(Self {
            upper_handles: counter.count,
            revision: Some(revision),
            growth,
            quote: PreparedRootQuote {
                raw_retained_bytes,
                retained_claims,
            },
        })
    }

    pub const fn quote(&self) -> PreparedRootQuote {
        self.quote
    }

    pub fn prepare<T: Trace + ?Sized>(
        self,
        source: &T,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<PreparedRootLease, RootAdmissionError> {
        self.try_prepare(
            |tracer| {
                source.trace(tracer);
                Ok::<_, std::convert::Infallible>(())
            },
            permit,
        )
        .map_err(trace_error::infallible_source)
    }

    /// Prepare from the same protected source used for planning. A scratch,
    /// cancellation, or source error is returned before registry publication.
    pub fn try_prepare<E>(
        self,
        trace: impl FnMut(&mut dyn Tracer) -> Result<(), E>,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<PreparedRootLease, RootTraceFailure<E>> {
        self.prepare_with_policy(trace, PreparationPolicy::Admitted(permit))
    }

    /// Execute the same transactional plan under an explicitly unbounded
    /// allocation policy. This is the compatibility path for runtimes whose
    /// canonical allocation policy is unbounded, not a fallback after a
    /// finite admission error.
    pub fn prepare_unbounded<T: Trace + ?Sized>(
        self,
        source: &T,
    ) -> Result<PreparedRootLease, RootAdmissionError> {
        self.prepare_with_policy(
            |tracer| {
                source.trace(tracer);
                Ok::<_, std::convert::Infallible>(())
            },
            PreparationPolicy::Unbounded,
        )
        .map_err(trace_error::infallible_source)
    }

    fn prepare_with_policy<E>(
        self,
        mut trace: impl FnMut(&mut dyn Tracer) -> Result<(), E>,
        policy: PreparationPolicy<'_>,
    ) -> Result<PreparedRootLease, RootTraceFailure<E>> {
        let lease_bytes = ExactShared::<RootLeasePayload>::requested_bytes()
            .checked_add(handle_layout(self.upper_handles)?.size())
            .ok_or(RootAdmissionError::LayoutOverflow)?;
        let lease_claim = if self.upper_handles == 0 {
            None
        } else {
            policy.reserve(lease_bytes)?
        };
        let mut handles = ExactHandleBuffer::new(self.upper_handles)?;
        let mut tracer = FillingTracer {
            handles: &mut handles,
            changed: false,
            seen: 0,
        };
        trace(&mut tracer).map_err(RootTraceFailure::Source)?;
        if tracer.changed || tracer.seen != self.upper_handles {
            return Err(RootAdmissionError::TraceChanged.into());
        }
        if self.upper_handles == 0 {
            return Ok(PreparedRootLease::empty());
        }
        handles.sort_dedup();

        let mut replacement = if let Some(growth) = self.growth {
            let allocator = match policy.reserve(growth.retained_bytes)? {
                Some(claim) => RegistryAllocator::admitted(growth.layout, claim)
                    .map_err(|_| RootAdmissionError::AllocationFailed)?,
                None => RegistryAllocator::unbounded(growth.layout),
            };
            Some(
                empty_map(growth.capacity, allocator)
                    .map_err(|_| RootAdmissionError::AllocationFailed)?,
            )
        } else {
            None
        };

        let payload = RootLeasePayload {
            handles: handles.into_vec(),
            armed: AtomicBool::new(false),
            _claim: lease_claim,
        };
        let owner =
            ExactShared::try_new(payload).map_err(|_| RootAdmissionError::AllocationFailed)?;

        let admission = ROOT_ADMISSION.write();
        let mut registry = GC.root_ptrs.lock();
        registry.validate_plan(
            self.revision
                .expect("nonempty root plan has a registry revision"),
        )?;
        for handle in &owner.get().handles {
            GC.validate_live_handle(handle)
                .map_err(|_| RootAdmissionError::InvalidHandle)?;
            registry.validate_increment(*handle)?;
        }
        if let Some(map) = replacement.as_mut() {
            for (handle, count) in registry.iter() {
                map.insert(*handle, *count);
            }
            for handle in &owner.get().handles {
                increment_map(map, *handle)?;
            }
        }
        let retired = registry.commit_prepared(&owner.get().handles, replacement);
        owner.get().armed.store(true, Ordering::Release);
        drop(registry);
        drop(admission);
        drop(retired);
        Ok(PreparedRootLease(Some(owner)))
    }
}

#[derive(Clone, Copy)]
enum PreparationPolicy<'a> {
    Admitted(&'a dyn ExactHostAllocationPermit),
    Unbounded,
}

impl PreparationPolicy<'_> {
    fn reserve(self, bytes: usize) -> Result<Option<ExactHostAllocationClaim>, RootAdmissionError> {
        match self {
            Self::Admitted(permit) => permit
                .reserve_exact_claim(bytes)
                .map(Some)
                .map_err(|_| RootAdmissionError::AllocationDenied),
            Self::Unbounded => Ok(None),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PreparedRootLease(Option<ExactShared<RootLeasePayload>>);

impl PreparedRootLease {
    fn empty() -> Self {
        Self(None)
    }

    /// Convenience for an explicitly unbounded allocation policy. The source
    /// is traced twice synchronously and must expose a stable graph across
    /// both passes.
    pub fn for_traced_unbounded<T: Trace + ?Sized>(source: &T) -> Result<Self, RootAdmissionError> {
        PreparedRootPlan::for_traced(source)?.prepare_unbounded(source)
    }

    pub fn handles(&self) -> &[GcHandle] {
        self.0
            .as_ref()
            .map_or(&[], |owner| owner.get().handles.as_slice())
    }
}

#[derive(Debug)]
struct RootLeasePayload {
    handles: Vec<GcHandle>,
    armed: AtomicBool,
    _claim: Option<ExactHostAllocationClaim>,
}

impl Drop for RootLeasePayload {
    fn drop(&mut self) {
        if self.armed.load(Ordering::Acquire) {
            GC.remove_root_batch(&self.handles);
        }
    }
}

struct CountingTracer {
    count: usize,
    overflow: bool,
}
impl Tracer for CountingTracer {
    fn mark(&mut self, _handle: GcHandle) {
        match self.count.checked_add(1) {
            Some(count) => self.count = count,
            None => self.overflow = true,
        }
    }
}

struct FillingTracer<'a> {
    handles: &'a mut ExactHandleBuffer,
    changed: bool,
    seen: usize,
}
impl Tracer for FillingTracer<'_> {
    fn mark(&mut self, handle: GcHandle) {
        self.seen = self.seen.saturating_add(1);
        if self.handles.push(handle).is_err() {
            self.changed = true;
        }
    }
}

struct ExactHandleBuffer {
    values: Vec<GcHandle>,
    layout: Layout,
}
impl ExactHandleBuffer {
    fn new(capacity: usize) -> Result<Self, RootAdmissionError> {
        let layout = handle_layout(capacity)?;
        if capacity == 0 {
            return Ok(Self {
                values: Vec::new(),
                layout,
            });
        }
        let pointer =
            NonNull::new(unsafe { alloc(layout) }).ok_or(RootAdmissionError::AllocationFailed)?;
        let values =
            unsafe { Vec::from_raw_parts(pointer.cast::<GcHandle>().as_ptr(), 0, capacity) };
        Ok(Self { values, layout })
    }
    fn push(&mut self, handle: GcHandle) -> Result<(), ()> {
        if self.values.len() == self.values.capacity() {
            return Err(());
        }
        self.values.push(handle);
        Ok(())
    }
    fn sort_dedup(&mut self) {
        self.values
            .sort_unstable_by_key(|handle| (handle.addr(), handle.epoch()));
        self.values.dedup();
    }
    fn into_vec(mut self) -> Vec<GcHandle> {
        let values = std::mem::take(&mut self.values);
        std::mem::forget(self);
        values
    }
}
impl Drop for ExactHandleBuffer {
    fn drop(&mut self) {
        if self.values.capacity() != 0 {
            let mut values = std::mem::take(&mut self.values);
            values.clear();
            let pointer = values.as_mut_ptr().cast::<u8>();
            std::mem::forget(values);
            unsafe { dealloc(pointer, self.layout) };
        }
    }
}

fn handle_layout(capacity: usize) -> Result<Layout, RootAdmissionError> {
    Layout::array::<GcHandle>(capacity).map_err(|_| RootAdmissionError::LayoutOverflow)
}
