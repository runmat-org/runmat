//! Upper bounds on the heap bytes that standard containers request for their contents.
//!
//! Meshing stages charge these bounds against the run's memory ceiling before or while their
//! working sets grow. Each bound counts requested allocation sizes (the bytes the global allocator
//! is asked for), on the target the code is compiled for; allocator headers and fragmentation are
//! the host's to cover in the ceiling it admits.

use std::mem::{size_of, MaybeUninit};

/// Bytes of a vector's heap buffer with `capacity` elements.
pub const fn vec_bytes<T>(capacity: usize) -> u64 {
    (capacity as u64).saturating_mul(size_of::<T>() as u64)
}

/// Bytes of one `Arc<T>` allocation: the value and its two reference counts.
pub const fn arc_bytes<T>() -> u64 {
    (size_of::<T>() + 2 * size_of::<usize>()) as u64
}

/// Bytes of one `Box<T>` allocation.
pub const fn box_bytes<T>() -> u64 {
    size_of::<T>() as u64
}

// Layout mirrors of the standard library's B-tree nodes (B = 6: eleven slots per node, twelve
// edges per internal node). `repr(C)` keeps the declared order, so a mirror is never smaller than
// the node it mirrors.
const BTREE_CAPACITY: usize = 11;
/// A non-root node holds at least B - 1 = 5 entries; a non-root internal node has at least
/// B = 6 children.
const BTREE_MINIMUM_ENTRIES: u64 = 5;

#[repr(C)]
struct LeafMirror<K, V> {
    parent: *const (),
    keys: [MaybeUninit<K>; BTREE_CAPACITY],
    values: [MaybeUninit<V>; BTREE_CAPACITY],
    parent_index: u16,
    length: u16,
}

#[repr(C)]
struct InternalMirror<K, V> {
    leaf: LeafMirror<K, V>,
    edges: [*const (); BTREE_CAPACITY + 1],
}

/// Upper bound on the heap bytes of a `BTreeMap<K, V>` (or a `BTreeSet<K>` with `V = ()`)
/// holding `length` entries, at any history of inserts and removals. An empty map that never
/// allocated is overcounted by one root leaf.
///
/// Every non-root leaf holds at least five entries, so there are at most `length / 5 + 1`
/// leaves. Every non-root internal node has at least six children, so each internal level has at
/// most a sixth of the nodes below it plus a root: at most `leaves / 5 + 1` internal nodes in all.
pub fn btree_bytes<K, V>(length: usize) -> u64 {
    // A map emptied by removals keeps its root leaf, so even zero entries count one leaf.
    let length = length as u64;
    let leaves = length / BTREE_MINIMUM_ENTRIES + 1;
    let internals = leaves / BTREE_MINIMUM_ENTRIES + 1;
    leaves
        .saturating_mul(size_of::<LeafMirror<K, V>>() as u64)
        .saturating_add(internals.saturating_mul(size_of::<InternalMirror<K, V>>() as u64))
}

/// Upper bound on the heap bytes of `maps` separate `BTreeMap<K, V>`s (or `BTreeSet<K>`s with
/// `V = ()`) that hold `entries` entries between them, however the entries are distributed.
///
/// Each map is bounded as in [`btree_bytes`]; summing those per-map node counts gives at most
/// `entries / 5 + maps` leaves and at most `leaves / 5 + maps` internal nodes in all.
pub fn btree_family_bytes<K, V>(maps: usize, entries: usize) -> u64 {
    let maps = maps as u64;
    let leaves = (entries as u64 / BTREE_MINIMUM_ENTRIES).saturating_add(maps);
    let internals = (leaves / BTREE_MINIMUM_ENTRIES).saturating_add(maps);
    leaves
        .saturating_mul(size_of::<LeafMirror<K, V>>() as u64)
        .saturating_add(internals.saturating_mul(size_of::<InternalMirror<K, V>>() as u64))
}
