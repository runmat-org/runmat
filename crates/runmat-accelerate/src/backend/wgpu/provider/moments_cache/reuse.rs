use std::collections::{HashMap, HashSet, VecDeque};

use runmat_accelerate_api::allocation::{
    AllocationIdentity, GpuTensorHandle, TensorViewIdentity, WeakGpuTensorHandle,
};

use super::{bounded::BoundedMoments, CacheBudget};

pub(super) type MomentsKey = (TensorViewIdentity, Vec<usize>);
type StrongMoments = BoundedMoments<MomentsKey, GpuTensorHandle, AllocationIdentity>;

struct SquareSource {
    base: WeakGpuTensorHandle,
    allocations: [AllocationIdentity; 2],
}

/// Both moment payloads and the adjacent weak square provenance have the same
/// provider-owned entry ceiling. The GPU byte ceiling applies only to retained
/// moment output allocations, independently of the adapter's buffer pool.
pub(in super::super) struct ReductionReuseCache {
    moments: StrongMoments,
    squares: HashMap<TensorViewIdentity, SquareSource>,
    square_oldest_first: VecDeque<TensorViewIdentity>,
    squares_by_allocation: HashMap<AllocationIdentity, HashSet<TensorViewIdentity>>,
    max_square_entries: usize,
}

impl ReductionReuseCache {
    pub(in super::super) fn new(budget: CacheBudget) -> Self {
        Self {
            moments: StrongMoments::new(budget),
            squares: HashMap::new(),
            square_oldest_first: VecDeque::new(),
            squares_by_allocation: HashMap::new(),
            max_square_entries: if budget.max_bytes == 0 {
                0
            } else {
                budget.max_entries
            },
        }
    }

    pub(super) fn square_source(
        &mut self,
        square: TensorViewIdentity,
    ) -> Option<WeakGpuTensorHandle> {
        let base = self.squares.get(&square)?.base.clone();
        self.square_oldest_first
            .retain(|candidate| *candidate != square);
        self.square_oldest_first.push_back(square);
        Some(base)
    }

    pub(super) fn record_square(
        &mut self,
        square: TensorViewIdentity,
        square_allocation: AllocationIdentity,
        base: &GpuTensorHandle,
    ) {
        self.remove_square(square);
        if self.max_square_entries == 0 {
            return;
        }
        while self.squares.len() >= self.max_square_entries {
            let Some(oldest) = self.square_oldest_first.front().copied() else {
                break;
            };
            self.remove_square(oldest);
        }
        let allocations = [square_allocation, base.allocation_identity()];
        for allocation in allocations {
            self.squares_by_allocation
                .entry(allocation)
                .or_default()
                .insert(square);
        }
        self.squares.insert(
            square,
            SquareSource {
                base: base.downgrade(),
                allocations,
            },
        );
        self.square_oldest_first.push_back(square);
    }

    pub(in super::super) fn moment_pair(
        &mut self,
        key: &MomentsKey,
    ) -> Option<(GpuTensorHandle, GpuTensorHandle)> {
        self.moments.get(key)
    }

    pub(super) fn record_moments(
        &mut self,
        key: MomentsKey,
        source: &GpuTensorHandle,
        mean: GpuTensorHandle,
        ex2: GpuTensorHandle,
        bytes: u64,
    ) -> Vec<GpuTensorHandle> {
        let allocations = [
            source.allocation_identity(),
            mean.allocation_identity(),
            ex2.allocation_identity(),
        ];
        let (_, detached) =
            self.moments
                .insert(key, allocations, Some(source.downgrade()), mean, ex2, bytes);
        detached
            .into_iter()
            .flat_map(|(mean, ex2)| [mean, ex2])
            .collect()
    }

    pub(super) fn source_views(&self) -> Vec<(MomentsKey, WeakGpuTensorHandle)> {
        self.moments.source_views()
    }

    pub(super) fn remove_stale_sources(&mut self, keys: &[MomentsKey]) -> Vec<GpuTensorHandle> {
        self.moments
            .remove_keys(keys)
            .into_iter()
            .flat_map(|(mean, ex2)| [mean, ex2])
            .collect()
    }

    /// Poison recovery discards all optional cache state. Strong handles are
    /// returned for release after the mutex is left; the poison flag remains.
    pub(in super::super) fn detach_all(&mut self) -> Vec<GpuTensorHandle> {
        self.squares.clear();
        self.square_oldest_first.clear();
        self.squares_by_allocation.clear();
        self.moments
            .detach_all()
            .into_iter()
            .flat_map(|(mean, ex2)| [mean, ex2])
            .collect()
    }

    /// Detach under the cache mutex; the physical owner drains the returned
    /// handles iteratively after it releases that mutex and its record mutex.
    pub(in super::super) fn retire_allocation(
        &mut self,
        allocation: AllocationIdentity,
    ) -> Vec<GpuTensorHandle> {
        let keys = self
            .squares_by_allocation
            .get(&allocation)
            .cloned()
            .unwrap_or_default();
        for key in keys {
            self.remove_square(key);
        }
        self.moments
            .retire(&allocation)
            .into_iter()
            .flat_map(|(mean, ex2)| [mean, ex2])
            .collect()
    }

    fn remove_square(&mut self, square: TensorViewIdentity) {
        let Some(entry) = self.squares.remove(&square) else {
            return;
        };
        self.square_oldest_first
            .retain(|candidate| *candidate != square);
        for allocation in entry.allocations {
            let empty = if let Some(keys) = self.squares_by_allocation.get_mut(&allocation) {
                keys.remove(&square);
                keys.is_empty()
            } else {
                false
            };
            if empty {
                self.squares_by_allocation.remove(&allocation);
            }
        }
    }
}
