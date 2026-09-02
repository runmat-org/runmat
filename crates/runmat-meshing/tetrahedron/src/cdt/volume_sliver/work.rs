use runmat_meshing_core::MeshingCancellationSignal;

use super::{error, DelaunayVolumeSliverError, DelaunayVolumeSliverErrorKind};

/// One hard candidate-evaluation budget shared by every optimization strategy
/// in a pass. Strategy boundaries must not reset either work or cancellation.
pub(in crate::cdt) struct OptimizationWork<'a> {
    evaluations: u64,
    maximum_evaluations: u64,
    cancellation_check_interval: u64,
    cancellation: &'a dyn MeshingCancellationSignal,
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
        }
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
