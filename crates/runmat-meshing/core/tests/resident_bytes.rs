//! The resident-byte bounds cover what the global allocator is actually asked for.
use runmat_meshing_core::resident_bytes::{arc_bytes, btree_bytes, btree_family_bytes, vec_bytes};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};

struct ThreadCounting;
thread_local! {
    static LIVE: Cell<u64> = const { Cell::new(0) };
}
#[global_allocator]
static ALLOCATOR: ThreadCounting = ThreadCounting;
unsafe impl GlobalAlloc for ThreadCounting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = LIVE.try_with(|live| live.set(live.get().wrapping_add(layout.size() as u64)));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        let _ = LIVE.try_with(|live| live.set(live.get().wrapping_sub(layout.size() as u64)));
        unsafe { System.dealloc(pointer, layout) }
    }
}
fn live() -> u64 {
    LIVE.with(Cell::get)
}

#[test]
fn btree_bound_covers_sequential_random_and_drained_maps() {
    for length in [1usize, 5, 11, 12, 100, 1_000, 50_000] {
        let before = live();
        let mut map = BTreeMap::new();
        for index in 0..length {
            map.insert([index as u64; 4], [0u8; 40]);
        }
        let filled = live().wrapping_sub(before);
        assert!(filled <= btree_bytes::<[u64; 4], [u8; 40]>(map.len()));
        // Pseudo-random removals leave sparse nodes; the bound holds at every length.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        while map.len() > length / 3 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let key = [state % length as u64; 4];
            map.remove(&key);
            let held = live().wrapping_sub(before);
            let bound = btree_bytes::<[u64; 4], [u8; 40]>(map.len());
            assert!(
                held <= bound,
                "length {} held {held} bound {bound}",
                map.len()
            );
        }
        drop(map);
        let before = live();
        let mut set = BTreeSet::new();
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        for _ in 0..length {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            set.insert(state);
        }
        assert!(live().wrapping_sub(before) <= btree_bytes::<u64, ()>(set.len()));
    }
}

#[test]
fn vector_and_shared_allocation_bounds_match_requested_sizes() {
    let before = live();
    let values = Vec::<[u32; 4]>::with_capacity(37);
    assert_eq!(
        live().wrapping_sub(before),
        vec_bytes::<[u32; 4]>(values.capacity())
    );
    let before = live();
    let shared = std::sync::Arc::new([0u64; 9]);
    assert_eq!(live().wrapping_sub(before), arc_bytes::<[u64; 9]>());
    drop((values, shared));
}

#[test]
fn btree_family_bound_covers_uneven_distributions() {
    // Many small sets beside one large one: the shape of vertex stars and face-use tables.
    let before = live();
    let mut sets = vec![BTreeSet::new(); 1_000];
    let mut entries = 0;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for index in 0..60_000usize {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let set = if index % 3 == 0 {
            0
        } else {
            (state % 1_000) as usize
        };
        if sets[set].insert(state) {
            entries += 1;
        }
    }
    let held = live().wrapping_sub(before) - vec_bytes::<BTreeSet<u64>>(sets.capacity());
    let bound = btree_family_bytes::<u64, ()>(sets.len(), entries);
    assert!(held <= bound, "held {held} bound {bound}");
    drop(sets);
}
