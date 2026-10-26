//! Publish one collector allocation only after its actual root is admitted.
//! Collector page accounting is unchanged. The callback admits the existing
//! root registry/lease backing under its original operation, without an await
//! or an allocator/value borrow held across Runtime reentry.
use crate::{
    allocator::GcAllocationKind, retirement::RetiredValue, AllocationGate, GarbageCollector,
    GcError, GcHandle, PreparedRootLease, Value, GC,
};

#[derive(Debug)]
pub enum GcRootedAllocationError<E> {
    Gc(GcError),
    Admission(E),
}

#[derive(Debug)]
pub struct PreparedGcAllocation {
    _roots: PreparedRootLease,
    handle: GcHandle,
}
impl PreparedGcAllocation {
    pub fn handle(&self) -> GcHandle {
        self.handle
    }
}

/// The callback must prepare the single actual handle through the canonical
/// PreparedRootPlan. Its lease is checked before the allocation may escape.
/// Child roots must already be protected before entry. No callback may retain
/// a borrowed GC value or publish this provisional handle on failure.
pub fn gc_allocate_prepared_rooted<E>(
    value: Value,
    prepare: impl FnOnce(GcHandle) -> std::result::Result<PreparedRootLease, E>,
) -> std::result::Result<PreparedGcAllocation, GcRootedAllocationError<E>> {
    // The caller already roots the original children. Match ordinary GC's
    // opportunistic preflight before taking the no-collection allocation gate.
    if GC.should_collect() {
        let _ = GC.collect_minor();
    }
    let gate = AllocationGate::enter(&GC);
    // allocate_inner releases its allocator mutex before returning. The gate
    // merely suppresses collection while the original root owner is called.
    let (handle, should_collect) = GC
        .allocate_inner(value)
        .map_err(GcRootedAllocationError::Gc)?;
    let mut unpublished = Unpublished {
        gc: &GC,
        handle: Some(handle),
    };
    let roots = prepare(handle).map_err(GcRootedAllocationError::Admission)?;
    if roots.handles() != std::slice::from_ref(&handle) {
        // Retire any unexpected registrations before the provisional payload.
        drop(roots);
        return Err(GcRootedAllocationError::Gc(
            GcError::RootRegistrationFailed(
                "prepared allocation requires its one exact original handle".into(),
            ),
        ));
    }
    unpublished.handle = None;
    let result = PreparedGcAllocation {
        _roots: roots,
        handle,
    };
    // The exact new root must exist before collection can inspect the Value.
    // No allocator/root-registration lock survives the preparation callback.
    drop(gate);
    if should_collect {
        // On failure, result's lease retires normally. The live Value keeps
        // its actual payload claims until normal GC retirement; no refund is
        // fabricated just because this operation cannot return its handle.
        GC.collect_minor().map_err(GcRootedAllocationError::Gc)?;
    }
    Ok(result)
}

struct Unpublished<'a> {
    gc: &'a GarbageCollector,
    handle: Option<GcHandle>,
}
impl Drop for Unpublished<'_> {
    fn drop(&mut self) {
        let Some(handle) = self.handle.take() else {
            return;
        };
        let retired = {
            let mut allocator = self.gc.allocator.lock();
            if !allocator.is_live_handle(&handle)
                || allocator.allocation_kind(&handle) != Some(GcAllocationKind::Value)
            {
                None
            } else {
                let pointer = handle.addr() as *const u8;
                if allocator.note_value_dropped(pointer) {
                    // SAFETY: this still-unpublished initialized Value was
                    // removed from the live set under the allocator mutex;
                    // no reuse can occur before the payload moves out.
                    Some(unsafe {
                        RetiredValue::take(pointer, Some(handle), GcAllocationKind::Value)
                    })
                } else {
                    None
                }
            }
        };
        if retired.is_some() {
            self.gc
                .stats
                .record_deallocation(std::mem::size_of::<Value>());
        }
        // Claims/finalizers may reenter Runtime/GC. No allocator, registry,
        // value-access, or collector lock remains held during destruction.
        drop(retired);
    }
}
