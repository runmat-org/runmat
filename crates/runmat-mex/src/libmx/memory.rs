use std::alloc::{alloc, alloc_zeroed, dealloc, realloc, Layout};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use runmat_value::{AdoptedHostAllocation, HostAllocationProvenance, HostAllocationRelease};

const MEX_ALLOCATOR_ID: &str = "runmat.mex.host.v1";
const MEX_ALLOCATION_ALIGNMENT: usize = std::mem::align_of::<u128>();

#[derive(Debug, Clone, Copy)]
struct AllocationRecord {
    byte_length: usize,
    persistent: bool,
}

#[derive(Debug, Default)]
pub(super) struct MexMemoryRegistry {
    allocations: HashMap<usize, AllocationRecord>,
}

#[derive(Debug)]
struct MexAllocationRelease {
    layout: Layout,
}

impl HostAllocationRelease for MexAllocationRelease {
    unsafe fn release(&self, pointer: NonNull<u8>) {
        // SAFETY: adoption records the exact layout used by the host allocator,
        // and ownership reaches this callback once.
        unsafe { dealloc(pointer.as_ptr(), self.layout) };
    }
}

impl MexMemoryRegistry {
    pub(super) fn allocate(&mut self, byte_length: usize, zeroed: bool) -> *mut c_void {
        let allocation_size = byte_length.max(1);
        let layout = allocation_layout(allocation_size);
        let pointer = if zeroed {
            // SAFETY: the validated layout is nonzero and has fixed alignment.
            unsafe { alloc_zeroed(layout) }
        } else {
            // SAFETY: the validated layout is nonzero and has fixed alignment.
            unsafe { alloc(layout) }
        };
        let Some(pointer) = NonNull::new(pointer) else {
            return std::ptr::null_mut();
        };
        self.allocations.insert(
            pointer.as_ptr() as usize,
            AllocationRecord {
                byte_length: allocation_size,
                persistent: false,
            },
        );
        pointer.as_ptr().cast()
    }

    pub(super) fn reallocate(&mut self, pointer: *mut c_void, byte_length: usize) -> *mut c_void {
        if pointer.is_null() {
            return self.allocate(byte_length, false);
        }
        let Some(record) = self.allocations.remove(&(pointer as usize)) else {
            return std::ptr::null_mut();
        };
        let allocation_size = byte_length.max(1);
        let old_layout = allocation_layout(record.byte_length);
        // SAFETY: the registry proves this live pointer came from the host
        // allocator with `old_layout` and has not been transferred or freed.
        let replacement = unsafe { realloc(pointer.cast(), old_layout, allocation_size) };
        let Some(replacement) = NonNull::new(replacement) else {
            self.allocations.insert(pointer as usize, record);
            return std::ptr::null_mut();
        };
        self.allocations.insert(
            replacement.as_ptr() as usize,
            AllocationRecord {
                byte_length: allocation_size,
                persistent: record.persistent,
            },
        );
        replacement.as_ptr().cast()
    }

    pub(super) fn free(&mut self, pointer: *mut c_void) -> bool {
        if pointer.is_null() {
            return true;
        }
        let Some(record) = self.allocations.remove(&(pointer as usize)) else {
            return false;
        };
        // SAFETY: removal proves exclusive release ownership.
        unsafe { dealloc(pointer.cast(), allocation_layout(record.byte_length)) };
        true
    }

    pub(super) fn make_persistent(&mut self, pointer: *mut c_void) -> bool {
        let Some(record) = self.allocations.get_mut(&(pointer as usize)) else {
            return false;
        };
        record.persistent = true;
        true
    }

    pub(super) fn take_for_adoption(
        &mut self,
        pointer: *mut c_void,
    ) -> Option<AdoptedHostAllocation> {
        let pointer = NonNull::new(pointer)?.cast::<u8>();
        let record = self.allocations.remove(&(pointer.as_ptr() as usize))?;
        let provenance = HostAllocationProvenance {
            allocator: MEX_ALLOCATOR_ID.into(),
            byte_length: record.byte_length,
            capacity_bytes: record.byte_length,
            alignment: MEX_ALLOCATION_ALIGNMENT,
        };
        // SAFETY: registry removal transfers this exact host allocation and its
        // sole release authority to the canonical value owner.
        Some(unsafe {
            AdoptedHostAllocation::new(
                pointer,
                provenance,
                Arc::new(MexAllocationRelease {
                    layout: allocation_layout(record.byte_length),
                }),
            )
        })
    }

    pub(super) fn finish_call(&mut self) {
        let temporary = self
            .allocations
            .iter()
            .filter_map(|(&pointer, record)| (!record.persistent).then_some(pointer))
            .collect::<Vec<_>>();
        for pointer in temporary {
            let _ = self.free(pointer as *mut c_void);
        }
    }
}

fn allocation_layout(byte_length: usize) -> Layout {
    Layout::from_size_align(byte_length.max(1), MEX_ALLOCATION_ALIGNMENT)
        .expect("fixed MEX host allocation alignment is valid")
}

impl Drop for MexMemoryRegistry {
    fn drop(&mut self) {
        let pointers = self.allocations.keys().copied().collect::<Vec<_>>();
        for pointer in pointers {
            let _ = self.free(pointer as *mut c_void);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adoption_transfers_the_registered_release_authority() {
        let mut registry = MexMemoryRegistry::default();
        let pointer = registry.allocate(2 * std::mem::size_of::<f64>(), true);
        assert!(!pointer.is_null());
        let allocation = registry.take_for_adoption(pointer).unwrap();
        assert!(registry.allocations.is_empty());
        assert_eq!(allocation.pointer().as_ptr().cast::<c_void>(), pointer);
        assert_eq!(
            allocation.provenance().byte_length,
            2 * std::mem::size_of::<f64>()
        );
        drop(allocation);
    }

    #[test]
    fn call_cleanup_retains_only_explicitly_persistent_memory() {
        let mut registry = MexMemoryRegistry::default();
        let temporary = registry.allocate(8, false);
        let persistent = registry.allocate(16, false);
        assert!(registry.make_persistent(persistent));
        registry.finish_call();
        assert!(!registry.allocations.contains_key(&(temporary as usize)));
        assert!(registry.allocations.contains_key(&(persistent as usize)));
        assert!(registry.free(persistent));
    }
}
