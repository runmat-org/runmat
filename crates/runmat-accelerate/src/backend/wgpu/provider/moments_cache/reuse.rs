use std::collections::{HashMap, VecDeque};

use runmat_accelerate_api::allocation::{
    AllocationIdentity, GpuTensorHandle, TensorViewIdentity, WeakGpuTensorHandle,
};

use super::{bounded::BoundedMoments, CacheBudget};
use runmat_execution::resource::RetainedHostAllocationClaim;
use std::sync::Arc;

pub(super) type MomentsKey = (TensorViewIdentity, Vec<usize>);
type StrongMoments = BoundedMoments<MomentsKey, GpuTensorHandle, AllocationIdentity>;

pub(super) fn cold_key_payload_bytes(dims: usize) -> Option<u64> {
    let inline = u64::try_from(StrongMoments::entry_inline_bytes()).ok()?;
    let dimensions = u64::try_from(dims)
        .ok()?
        .checked_mul(std::mem::size_of::<usize>() as u64)?
        // Entry and LRU each own one dimensions Vec. One additional cloned
        // key may coexist while updating LRU or retiring an allocation.
        .checked_mul(3)?;
    inline.checked_add(dimensions)
}

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
    max_square_entries: usize,
}

impl ReductionReuseCache {
    pub(in super::super) fn try_new(
        budget: CacheBudget,
    ) -> Result<Self, std::collections::TryReserveError> {
        let moments = StrongMoments::try_new(budget)?;
        let mut squares = HashMap::new();
        let entries_bound = if budget.max_bytes == 0 {
            0
        } else {
            budget.max_entries
        };
        squares.try_reserve(entries_bound)?;
        let mut square_oldest_first = VecDeque::new();
        square_oldest_first.try_reserve(entries_bound)?;
        Ok(Self {
            moments,
            squares,
            square_oldest_first,
            max_square_entries: if budget.max_bytes == 0 {
                0
            } else {
                budget.max_entries
            },
        })
    }

    #[cfg(test)]
    pub(in super::super) fn new(budget: CacheBudget) -> Self {
        Self::try_new(budget).expect("test cache capacity")
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
        metadata_claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
    ) -> Vec<GpuTensorHandle> {
        let allocations = [
            source.allocation_identity(),
            mean.allocation_identity(),
            ex2.allocation_identity(),
        ];
        let (_, detached) = self.moments.insert(super::bounded::MomentEntry {
            key,
            allocations,
            source_view: Some(source.downgrade()),
            mean,
            ex2,
            bytes,
            metadata_claim,
        });
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
        while let Some(key) = self
            .squares
            .iter()
            .find_map(|(key, entry)| entry.allocations.contains(&allocation).then_some(*key))
        {
            self.remove_square(key);
        }
        self.moments
            .retire(&allocation)
            .into_iter()
            .flat_map(|(mean, ex2)| [mean, ex2])
            .collect()
    }

    fn remove_square(&mut self, square: TensorViewIdentity) {
        if self.squares.remove(&square).is_none() {
            return;
        }
        self.square_oldest_first
            .retain(|candidate| *candidate != square);
    }
}
