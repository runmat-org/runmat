use runmat_meshing_core::MeshingCancellationSignal;

use super::{
    error, DelaunayVolumeRefinementCandidateError, DelaunayVolumeRefinementCandidateErrorKind,
    DelaunayVolumeRefinementCandidateOptions,
};

pub(super) struct CandidateWork<'a> {
    pub(super) options: DelaunayVolumeRefinementCandidateOptions,
    pub(super) cancellation: &'a dyn MeshingCancellationSignal,
    evaluations: u64,
}

impl<'a> CandidateWork<'a> {
    pub(super) fn new(
        options: DelaunayVolumeRefinementCandidateOptions,
        cancellation: &'a dyn MeshingCancellationSignal,
    ) -> Self {
        Self {
            options,
            cancellation,
            evaluations: 0,
        }
    }

    /// Construction currently charges its two stages before returning a point.
    /// Batch selection can hand off ready work before requesting a unit beyond
    /// the caller's unchanged evaluation ceiling.
    pub(super) fn can_construct(&self) -> bool {
        self.options
            .maximum_candidate_evaluations
            .saturating_sub(self.evaluations)
            >= 2
    }

    pub(super) fn evaluations(&self) -> u64 {
        self.evaluations
    }

    pub(super) fn evaluate(&mut self) -> Result<(), DelaunayVolumeRefinementCandidateError> {
        if self
            .evaluations
            .is_multiple_of(self.options.cancellation_check_interval)
            && self.cancellation.is_cancelled()
        {
            return Err(error(
                DelaunayVolumeRefinementCandidateErrorKind::Cancelled,
                "cancelled",
            ));
        }
        self.evaluations = self.evaluations.saturating_add(1);
        if self.evaluations > self.options.maximum_candidate_evaluations {
            return Err(error(
                DelaunayVolumeRefinementCandidateErrorKind::ResourceLimit,
                "candidate evaluation limit exceeded",
            ));
        }
        Ok(())
    }
}
