use runmat_meshing_core::MeshingCancellationSignal;

use super::{error, DelaunayVolumeSliverError, DelaunayVolumeSliverErrorKind};

/// One hard candidate-evaluation budget shared by every optimization strategy
/// in a pass. Strategy boundaries must not reset either work or cancellation.
pub(in crate::cdt) struct OptimizationWork<'a> {
    evaluations: u64,
    maximum_evaluations: u64,
    cancellation_check_interval: u64,
    cancellation: &'a dyn MeshingCancellationSignal,
    global_candidates: u64,
    global_node_inventory: u64,
    global_cell_inventory: u64,
}

impl<'a> OptimizationWork<'a> {
    pub(in crate::cdt) fn new(
        maximum_evaluations: u64,
        cancellation_check_interval: u64,
        cancellation: &'a dyn MeshingCancellationSignal,
    ) -> Self {
        Self {
            evaluations: 0,
            maximum_evaluations,
            cancellation_check_interval,
            cancellation,
            global_candidates: 0,
            global_node_inventory: 0,
            global_cell_inventory: 0,
        }
    }

    /// Input inventory offered to explicit complete candidates, not measured
    /// allocator bytes, primitive operations or a claim of localized work.
    pub(super) fn global_candidate(
        &mut self,
        nodes: usize,
        cells: usize,
    ) -> Result<(), DelaunayVolumeSliverError> {
        self.global_candidates = self.global_candidates.checked_add(1).ok_or_else(|| {
            error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                "global candidate count overflow",
            )
        })?;
        self.global_node_inventory = self
            .global_node_inventory
            .checked_add(nodes as u64)
            .ok_or_else(|| {
                error(
                    DelaunayVolumeSliverErrorKind::ResourceLimit,
                    "global node inventory overflow",
                )
            })?;
        self.global_cell_inventory = self
            .global_cell_inventory
            .checked_add(cells as u64)
            .ok_or_else(|| {
                error(
                    DelaunayVolumeSliverErrorKind::ResourceLimit,
                    "global cell inventory overflow",
                )
            })?;
        Ok(())
    }

    pub(in crate::cdt) fn evaluate(&mut self) -> Result<(), DelaunayVolumeSliverError> {
        if self.evaluations >= self.maximum_evaluations {
            return Err(error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                format!(
                    "volume optimization exhausted its shared budget of {} candidate evaluations",
                    self.maximum_evaluations
                ),
            ));
        }
        if self
            .evaluations
            .is_multiple_of(self.cancellation_check_interval)
            && self.cancellation.is_cancelled()
        {
            return Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                "volume optimization candidate search was cancelled",
            ));
        }
        self.evaluations += 1;
        Ok(())
    }
}

impl Drop for OptimizationWork<'_> {
    fn drop(&mut self) {
        tracing::info!(
            candidate_evaluations = self.evaluations,
            global_candidate_attempts = self.global_candidates,
            global_candidate_node_inventory = self.global_node_inventory,
            global_candidate_cell_inventory = self.global_cell_inventory,
            "volume optimization work inventory"
        );
    }
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::NeverCancelled;

    use super::*;

    #[test]
    fn candidate_budget_is_hard_and_shared_by_callers() {
        let mut work = OptimizationWork::new(2, 1, &NeverCancelled);
        work.evaluate().unwrap();
        work.evaluate().unwrap();
        assert_eq!(
            work.evaluate().unwrap_err().kind,
            DelaunayVolumeSliverErrorKind::ResourceLimit
        );
    }
}
