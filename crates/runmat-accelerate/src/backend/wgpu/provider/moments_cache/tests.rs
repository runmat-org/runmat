use super::bounded::BoundedMoments;
use super::*;
use runmat_accelerate_api::test_support::TestAllocationOwner;
use runmat_accelerate_api::{GpuTensorStorage, NumericElementType};
use std::sync::Arc;

type TestCache = BoundedMoments<(u64, Vec<usize>), Arc<u64>, u64>;
type TestMomentPair = (Arc<u64>, Arc<u64>);

fn insert(
    cache: &mut TestCache,
    view: u64,
    dims: &[usize],
    source: u64,
    bytes: u64,
) -> (bool, Vec<TestMomentPair>) {
    cache.insert(bounded::MomentEntry {
        key: (view, dims.to_vec()),
        allocations: [source, view + 100, view + 200],
        source_view: None,
        mean: Arc::new(view + 100),
        ex2: Arc::new(view + 200),
        bytes,
        metadata_claim: None,
    })
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn exact_view_dimension_hits_and_lru_eviction_respect_both_budgets() {
    let mut cache = TestCache::new(CacheBudget {
        max_entries: 2,
        max_bytes: 32,
    });
    assert!(insert(&mut cache, 1, &[0], 9, 16).0);
    assert!(insert(&mut cache, 2, &[0], 9, 16).0);
    let first = cache.get(&(1, vec![0])).expect("same view/dimension pair");
    assert_eq!((*first.0, *first.1), (101, 201));
    assert!(cache.get(&(1, vec![1])).is_none());
    let (retained, detached) = insert(&mut cache, 3, &[0], 8, 16);
    assert!(retained);
    assert_eq!(detached.len(), 1);
    assert_eq!((*detached[0].0, *detached[0].1), (102, 202));
    assert!(cache.get(&(2, vec![0])).is_none());
    assert_eq!(cache.retained(), (2, 32));
    drop(detached);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn zero_budget_and_oversized_pair_bypass_without_displacing_another_pair() {
    let mut disabled = TestCache::new(CacheBudget {
        max_entries: 0,
        max_bytes: 32,
    });
    let (retained, detached) = insert(&mut disabled, 1, &[0], 1, 16);
    assert!(!retained);
    assert_eq!(detached.len(), 1);
    assert_eq!(disabled.retained(), (0, 0));
    drop(detached);

    let mut cache = TestCache::new(CacheBudget {
        max_entries: 2,
        max_bytes: 32,
    });
    assert!(insert(&mut cache, 1, &[0], 1, 16).0);
    let (retained, detached) = insert(&mut cache, 2, &[0], 2, 33);
    assert!(!retained);
    assert_eq!(detached.len(), 1);
    assert_eq!(cache.retained(), (1, 16));
    assert!(cache.get(&(1, vec![0])).is_some());
    drop(detached);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn physical_retirement_purges_all_views_dimensions_and_output_aliases() {
    let mut cache = TestCache::new(CacheBudget {
        max_entries: 8,
        max_bytes: 128,
    });
    for (view, dims, source) in [(1, vec![0], 7), (2, vec![1], 7), (3, vec![0], 8)] {
        assert!(insert(&mut cache, view, &dims, source, 16).0);
    }
    let detached = cache.retire(&7);
    assert_eq!(detached.len(), 2);
    assert!(cache.get(&(1, vec![0])).is_none());
    assert!(cache.get(&(2, vec![1])).is_none());
    assert!(cache.get(&(3, vec![0])).is_some());
    drop(detached);
    assert_eq!(cache.retire(&103).len(), 1);
    assert_eq!(cache.retained(), (0, 0));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn long_replacement_chain_detaches_iteratively_without_unbounded_retention() {
    let mut cache = TestCache::new(CacheBudget {
        max_entries: 2,
        max_bytes: 32,
    });
    for view in 0..4096 {
        let (retained, detached) = insert(&mut cache, view, &[0], view, 16);
        assert!(retained);
        drop(detached);
        assert!(cache.retained().0 <= 2);
        assert!(cache.retained().1 <= 32);
    }
    assert_eq!(cache.retained(), (2, 32));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn square_provenance_is_weak_bounded_and_retired_by_canonical_allocation() {
    let owner = TestAllocationOwner::new();
    let base = owner
        .allocate_zeroed(vec![1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let first = owner
        .allocate_zeroed(vec![1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let second = owner
        .allocate_zeroed(vec![1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let mut cache = ReductionReuseCache::new(CacheBudget {
        max_entries: 1,
        max_bytes: 64,
    });
    cache.record_square(
        first.handle.view_identity(),
        first.handle.allocation_identity(),
        &base.handle,
    );
    assert_eq!(
        cache
            .square_source(first.handle.view_identity())
            .unwrap()
            .acquire()
            .unwrap()
            .view_identity(),
        base.handle.view_identity()
    );
    cache.record_square(
        second.handle.view_identity(),
        second.handle.allocation_identity(),
        &base.handle,
    );
    assert!(cache.square_source(first.handle.view_identity()).is_none());
    assert!(cache.square_source(second.handle.view_identity()).is_some());
    assert!(cache
        .retire_allocation(base.handle.allocation_identity())
        .is_empty());
    assert!(cache.square_source(second.handle.view_identity()).is_none());
    let mut disabled = ReductionReuseCache::new(CacheBudget {
        max_entries: 1,
        max_bytes: 0,
    });
    disabled.record_square(
        second.handle.view_identity(),
        second.handle.allocation_identity(),
        &base.handle,
    );
    assert!(disabled
        .square_source(second.handle.view_identity())
        .is_none());
    drop(base.handle);
    assert_eq!(
        base.probe.retirements(),
        1,
        "weak provenance did not retain base"
    );
}

#[path = "tests/cascade.rs"]
mod cascade;
