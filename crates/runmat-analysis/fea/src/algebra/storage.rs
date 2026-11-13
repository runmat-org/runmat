//! Requested heap storage for the numerical layer's shared vector owners.
//!
//! The pinned Rust toolchain allocates an Arc's two atomic counters and Vec
//! descriptor separately from the Vec's element allocation. Count both, including
//! an empty shared Vec. Allocator bookkeeping and borrowed aliases are excluded.
//! The contact allocation witnesses compare these counts with actual requests.
use std::mem::size_of;
use std::sync::atomic::AtomicUsize;

pub(crate) fn shared_vector_storage_bytes<T>(capacity: usize) -> Option<u64> {
    let header = 2 * size_of::<AtomicUsize>() + size_of::<Vec<T>>();
    capacity
        .checked_mul(size_of::<T>())
        .and_then(|bytes| bytes.checked_add(header))
        .and_then(|bytes| u64::try_from(bytes).ok())
}
