//! Exact allocation owner for immutable root-registry generations.

use allocator_api2::alloc::{AllocError, Allocator};
use hashbrown::HashMap;
use runmat_execution::resource::ExactShared;
use runmat_value::ExactHostAllocationClaim;
use std::alloc::{alloc, dealloc, Layout};
use std::cell::Cell;
use std::collections::hash_map::RandomState;
use std::hash::Hash;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

struct AllocationHeader;

#[derive(Debug)]
struct AllocatorState {
    expected: Layout,
    allocated: AtomicBool,
    _claim: Option<ExactHostAllocationClaim>,
}

#[derive(Clone, Debug)]
pub(super) struct RegistryAllocator(ExactShared<AllocatorState>);

impl RegistryAllocator {
    pub(super) fn unbounded(expected: Layout) -> Self {
        Self(ExactShared::new(AllocatorState {
            expected,
            allocated: AtomicBool::new(false),
            _claim: None,
        }))
    }

    pub(super) fn admitted(expected: Layout, claim: ExactHostAllocationClaim) -> Result<Self, ()> {
        ExactShared::try_new(AllocatorState {
            expected,
            allocated: AtomicBool::new(false),
            _claim: Some(claim),
        })
        .map(Self)
        .map_err(|_| ())
    }

    pub(super) const fn owner_bytes() -> usize {
        ExactShared::<AllocatorState>::requested_bytes()
    }

    pub(super) fn block_bytes(layout: Layout) -> Result<usize, AllocError> {
        block_layout(layout).map(|(block, _)| block.size())
    }
}

fn block_layout(layout: Layout) -> Result<(Layout, usize), AllocError> {
    Layout::new::<AllocationHeader>()
        .extend(layout)
        .map_err(|_| AllocError)
}

// SAFETY: every generation performs one allocation. Its exact owner retains
// the claim until after hashbrown has released the table block.
unsafe impl Allocator for RegistryAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let state = self.0.get();
        if layout != state.expected || state.allocated.swap(true, Ordering::AcqRel) {
            return Err(AllocError);
        }
        let (block, offset) = block_layout(layout)?;
        let pointer = NonNull::new(unsafe { alloc(block) }).ok_or(AllocError)?;
        // SAFETY: `extend` supplies an aligned payload within this live block.
        let payload = unsafe { NonNull::new_unchecked(pointer.as_ptr().add(offset)) };
        Ok(NonNull::slice_from_raw_parts(payload, layout.size()))
    }

    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
        let (block, offset) = block_layout(layout)
            .expect("registry deallocation uses its admitted allocation layout");
        let base = unsafe { pointer.as_ptr().sub(offset) };
        unsafe { dealloc(base, block) };
    }
}

#[derive(Clone, Copy)]
struct LayoutProbe<'a>(&'a Cell<Option<Layout>>);

// SAFETY: the probe never returns an allocation or receives deallocation.
unsafe impl Allocator for LayoutProbe<'_> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.set(Some(layout));
        Err(AllocError)
    }

    unsafe fn deallocate(&self, _pointer: NonNull<u8>, _layout: Layout) {
        unreachable!("a registry layout probe never allocates")
    }
}

pub(super) fn table_layout<K: Eq + Hash, V>(capacity: usize) -> Result<Layout, ()> {
    let requested = Cell::new(None);
    let allocator = LayoutProbe(&requested);
    let mut table: HashMap<K, V, RandomState, LayoutProbe<'_>> =
        HashMap::with_hasher_in(RandomState::new(), allocator);
    let result = table.try_reserve(capacity);
    let layout = requested.get();
    if result.is_ok() || layout.is_none() {
        return Err(());
    }
    layout.ok_or(())
}

pub(super) type RootMap<K, V> = HashMap<K, V, RandomState, RegistryAllocator>;

pub(super) fn empty_map<K: Eq + Hash, V>(
    capacity: usize,
    allocator: RegistryAllocator,
) -> Result<RootMap<K, V>, ()> {
    let mut map = HashMap::with_hasher_in(RandomState::new(), allocator);
    map.try_reserve(capacity).map_err(|_| ())?;
    Ok(map)
}
