//! Exact admission for collection blocks, including their retained claim.
//!
//! Collections supply their actual allocation layout. This owner adds only
//! its known-layout header and alignment padding; table capacity and ordering
//! remain with the collection's domain.

use allocator_api2::alloc::{AllocError, Allocator};
use hashbrown::HashMap;
use runmat_execution::resource::ExactShared;
use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError,
};
use std::alloc::{alloc, dealloc, Layout};
use std::cell::{Cell, RefCell};
use std::collections::hash_map::RandomState;
use std::hash::Hash;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

use super::{HostAllocationError, HostAllocationPlan, HostOutputPermit};

struct AllocationHeader {
    claim: ManuallyDrop<Option<ExactHostAllocationClaim>>,
}

#[derive(Clone, Debug)]
pub struct AdmittedCollectionAllocator {
    permit: HostOutputPermit,
}

impl AdmittedCollectionAllocator {
    pub(crate) fn new(permit: HostOutputPermit) -> Self {
        Self { permit }
    }

    pub(crate) const fn retained_claim_count() -> usize {
        1
    }

    /// Requested block bytes; the permit's claim control is quoted separately.
    pub(crate) fn requested_bytes(layout: Layout) -> Result<usize, AllocError> {
        Self::block_layout(layout).map(|(block, _)| block.size())
    }

    fn block_layout(layout: Layout) -> Result<(Layout, usize), AllocError> {
        Layout::new::<AllocationHeader>()
            .extend(layout)
            .map_err(|_| AllocError)
    }
}

/// A long-lived collection can use a new, selected operation's output permit
/// for each growth. The gate contains no RuntimeContext or ledger authority:
/// each allocated block retains its own exact claim after the operation ends.
#[derive(Debug)]
struct OperationPermitGatePayload {
    current: RefCell<Option<OperationAllocationPolicy>>,
    _claim: Option<ExactHostAllocationClaim>,
}

#[derive(Clone, Debug)]
enum OperationAllocationPolicy {
    Admitted(HostOutputPermit),
    Unbounded,
}

#[derive(Clone, Debug)]
pub(crate) struct OperationPermitGate(ExactShared<OperationPermitGatePayload>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OperationPermitGateError {
    AlreadyInstalled,
    Borrowed,
}

pub(crate) struct InstalledCollectionPermit {
    gate: OperationPermitGate,
}

impl OperationPermitGate {
    pub(crate) fn retained_bytes() -> Result<u64, HostAllocationError> {
        HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(ExactShared::<OperationPermitGatePayload>::requested_bytes())
                .map_err(|_| HostAllocationError::Overflow)?,
            1,
        )
    }

    pub(crate) fn prepare(
        permit: &HostOutputPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let claim = permit
            .reserve_exact_claim(ExactShared::<OperationPermitGatePayload>::requested_bytes())?;
        ExactShared::try_new(OperationPermitGatePayload {
            current: RefCell::new(None),
            _claim: Some(claim),
        })
        .map(Self)
        .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
    }

    pub(crate) fn install(
        &self,
        permit: HostOutputPermit,
    ) -> Result<InstalledCollectionPermit, OperationPermitGateError> {
        self.install_policy(OperationAllocationPolicy::Admitted(permit))
    }

    pub(crate) fn unbounded() -> Self {
        Self(ExactShared::new(OperationPermitGatePayload {
            current: RefCell::new(None),
            _claim: None,
        }))
    }

    pub(crate) fn install_unbounded(
        &self,
    ) -> Result<InstalledCollectionPermit, OperationPermitGateError> {
        self.install_policy(OperationAllocationPolicy::Unbounded)
    }

    fn install_policy(
        &self,
        policy: OperationAllocationPolicy,
    ) -> Result<InstalledCollectionPermit, OperationPermitGateError> {
        let mut current = self
            .0
            .get()
            .current
            .try_borrow_mut()
            .map_err(|_| OperationPermitGateError::Borrowed)?;
        if current.is_some() {
            return Err(OperationPermitGateError::AlreadyInstalled);
        }
        *current = Some(policy);
        Ok(InstalledCollectionPermit { gate: self.clone() })
    }

    fn current_policy(&self) -> Result<OperationAllocationPolicy, AllocError> {
        self.0
            .get()
            .current
            .try_borrow()
            .map_err(|_| AllocError)?
            .as_ref()
            .cloned()
            .ok_or(AllocError)
    }
}

impl Drop for InstalledCollectionPermit {
    fn drop(&mut self) {
        let mut current = self
            .gate
            .0
            .get()
            .current
            .try_borrow_mut()
            .expect("collection allocation cannot retain a permit-gate borrow");
        current.take();
    }
}

#[derive(Clone, Debug)]
pub struct OperationCollectionAllocator {
    gate: OperationPermitGate,
}

impl OperationCollectionAllocator {
    pub(crate) fn new(gate: OperationPermitGate) -> Self {
        Self { gate }
    }
}

#[derive(Clone, Copy)]
struct HashTableLayoutProbe<'a>(&'a Cell<Option<Layout>>);

// SAFETY: no probe allocation succeeds or transfers ownership. The exact
// layout requested by the pinned hashbrown implementation is recorded.
unsafe impl Allocator for HashTableLayoutProbe<'_> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.set(Some(layout));
        Err(AllocError)
    }

    unsafe fn deallocate(&self, _pointer: NonNull<u8>, _layout: Layout) {
        unreachable!("a hash-table layout probe never allocates")
    }
}

pub fn hashbrown_table_layout<K: Eq + Hash, V>(
    capacity: usize,
) -> Result<Option<Layout>, HostAllocationError> {
    if capacity == 0 {
        return Ok(None);
    }
    let requested = Cell::new(None);
    let allocator = HashTableLayoutProbe(&requested);
    let mut table: HashMap<K, V, RandomState, HashTableLayoutProbe<'_>> =
        HashMap::with_hasher_in(RandomState::new(), allocator);
    let result = table.try_reserve(capacity);
    let layout = requested.get();
    if result.is_ok() || layout.is_none() {
        return Err(HostAllocationError::Overflow);
    }
    Ok(layout)
}

// SAFETY: the gate is consulted only for new allocation. The block's own
// header retains its claim and deallocation needs no live operation permit.
unsafe impl Allocator for OperationCollectionAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let policy = self.gate.current_policy()?;
        allocate_collection_block(&policy, layout)
    }

    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
        unsafe { deallocate_collection_block(pointer, layout) }
    }
}

// SAFETY: each successful allocation owns a distinct exact-layout block.
// Moving/cloning this allocator moves/clones only the fenced permit; block
// lifetime and refund authority reside in the allocation's own header.
unsafe impl Allocator for AdmittedCollectionAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        allocate_collection_block(
            &OperationAllocationPolicy::Admitted(self.permit.clone()),
            layout,
        )
    }

    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
        unsafe { deallocate_collection_block(pointer, layout) }
    }
}

fn allocate_collection_block(
    policy: &OperationAllocationPolicy,
    layout: Layout,
) -> Result<NonNull<[u8]>, AllocError> {
    let (block, offset) = AdmittedCollectionAllocator::block_layout(layout)?;
    let claim = match policy {
        OperationAllocationPolicy::Admitted(permit) => Some(
            permit
                .reserve_exact_claim(block.size())
                .map_err(|_| AllocError)?,
        ),
        OperationAllocationPolicy::Unbounded => None,
    };
    // SAFETY: block contains a nonzero header and has a valid Layout.
    let Some(pointer) = NonNull::new(unsafe { alloc(block) }) else {
        // No physical block exists; the stack claim refunds on return.
        return Err(AllocError);
    };
    // SAFETY: initialize the header once; extend guarantees the payload
    // offset satisfies the requested alignment, including size-zero data.
    let payload = unsafe {
        pointer
            .cast::<AllocationHeader>()
            .as_ptr()
            .write(AllocationHeader {
                claim: ManuallyDrop::new(claim),
            });
        NonNull::new_unchecked(pointer.as_ptr().add(offset))
    };
    Ok(NonNull::slice_from_raw_parts(payload, layout.size()))
}

unsafe fn deallocate_collection_block(pointer: NonNull<u8>, layout: Layout) {
    let (block, offset) = AdmittedCollectionAllocator::block_layout(layout)
        .expect("deallocation uses the successful allocation's exact layout");
    // SAFETY: pointer/layout identify this allocator's live payload.
    // Allocation returned exactly layout.size(), so no larger fitting
    // layout may be supplied by the caller.
    let header = unsafe { pointer.as_ptr().sub(offset).cast::<AllocationHeader>() };
    // SAFETY: the unique block is being consumed; take its sole claim
    // before freeing the header which contains that claim.
    let claim = unsafe { ManuallyDrop::take(&mut (*header).claim) };
    // SAFETY: header is the original global pointer, with its exact layout.
    unsafe { dealloc(header.cast::<u8>(), block) };
    drop(claim);
}
