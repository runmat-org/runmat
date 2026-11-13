use runmat_meshing_core::MeshingCancellationSignal;

use super::{
    error, resource, DelaunayFacetRecoveryError, DelaunayFacetRecoveryErrorKind,
    DelaunayFacetRecoveryOptions,
};

/// Work accounting for facet recovery and its terminal verification.
///
/// Every counter bounds the work of one facet constraint: `begin_constraint` starts a fresh
/// account. A facet whose support construction, topology search, flips or cavity work runs past
/// its bound is a runaway local search and fails with `ResourceLimit` at that constraint. Total
/// work therefore grows with the number of facets the surface hands to the volume stage, and the
/// run's admitted time budget (cancellation) bounds the whole pass. A single account shared by
/// all facets would instead fail a large, healthy boundary at whichever facet happened to spend
/// the last unit, with no local defect behind it.
pub(super) struct FacetRecoveryWork<'a> {
    pub(super) options: DelaunayFacetRecoveryOptions,
    pub(super) cancellation: &'a dyn MeshingCancellationSignal,
    search_steps: u64,
    flip_attempts: u64,
    support_steps: u64,
    cavity_steps: u64,
    cavity_apex_attempts: u64,
    cavity_steiner_nodes: u64,
    cavity_steiner_insertion_attempts: u64,
    last_rejection: Option<String>,
}

impl<'a> FacetRecoveryWork<'a> {
    pub(super) fn new(
        options: DelaunayFacetRecoveryOptions,
        cancellation: &'a dyn MeshingCancellationSignal,
    ) -> Self {
        Self {
            options,
            cancellation,
            search_steps: 0,
            flip_attempts: 0,
            support_steps: 0,
            cavity_steps: 0,
            cavity_apex_attempts: 0,
            cavity_steiner_nodes: 0,
            cavity_steiner_insertion_attempts: 0,
            last_rejection: None,
        }
    }

    pub(super) fn clear_rejection(&mut self) {
        self.last_rejection = None;
    }

    pub(super) fn begin_constraint(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        // Local work limits restart for each facet. Cancellation also has a
        // boundary checkpoint, so short facets cannot postpone every probe.
        self.poll_cancellation(constraint_index)?;
        self.search_steps = 0;
        self.flip_attempts = 0;
        self.support_steps = 0;
        self.cavity_steps = 0;
        self.cavity_apex_attempts = 0;
        self.cavity_steiner_nodes = 0;
        self.cavity_steiner_insertion_attempts = 0;
        self.last_rejection = None;
        Ok(())
    }

    pub(super) fn reject(&mut self, reason: impl Into<String>) {
        self.last_rejection = Some(reason.into());
    }

    pub(super) fn last_rejection(&self) -> Option<&str> {
        self.last_rejection.as_deref()
    }

    pub(super) fn search_step(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        self.search_steps += 1;
        if self.search_steps > self.options.maximum_search_steps {
            return Err(resource(
                constraint_index,
                "facet topology-search limit exceeded for one facet",
            ));
        }
        self.check_cancelled(constraint_index, self.search_steps)
    }

    pub(super) fn flip_attempt(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        self.flip_attempts += 1;
        if self.flip_attempts > self.options.maximum_flip_attempts {
            return Err(resource(
                constraint_index,
                "facet edge-flip attempt limit exceeded for one facet",
            ));
        }
        self.check_cancelled(constraint_index, self.flip_attempts)
    }

    pub(super) fn support_step(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        self.support_steps += 1;
        if self.support_steps > self.options.maximum_support_steps {
            return Err(resource(
                constraint_index,
                "facet support-construction limit exceeded for one facet",
            ));
        }
        self.check_cancelled(constraint_index, self.support_steps)
    }

    pub(super) fn cavity_step(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        self.cavity_steps += 1;
        if self.cavity_steps > self.options.maximum_cavity_steps {
            return Err(resource(
                constraint_index,
                "facet cavity work limit exceeded for one facet",
            ));
        }
        self.check_cancelled(constraint_index, self.cavity_steps)
    }

    pub(super) fn cavity_apex_attempt(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        self.cavity_apex_attempts += 1;
        if self.cavity_apex_attempts > self.options.maximum_cavity_apex_attempts {
            return Err(resource(
                constraint_index,
                "facet cavity apex-attempt limit exceeded",
            ));
        }
        self.check_cancelled(constraint_index, self.cavity_apex_attempts)
    }

    pub(super) fn cavity_steiner_node(
        &mut self,
        constraint_index: u32,
    ) -> Result<u64, DelaunayFacetRecoveryError> {
        self.cavity_steiner_nodes += 1;
        if self.cavity_steiner_nodes > self.options.maximum_cavity_steiner_nodes {
            return Err(resource(
                constraint_index,
                "facet cavity Steiner-node limit exceeded",
            ));
        }
        self.check_cancelled(constraint_index, self.cavity_steiner_nodes)?;
        Ok(self.cavity_steiner_nodes)
    }

    pub(super) fn cavity_steiner_insertion_attempt(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        self.cavity_steiner_insertion_attempts += 1;
        if self.cavity_steiner_insertion_attempts > self.options.maximum_cavity_steiner_candidates {
            return Err(resource(
                constraint_index,
                "facet cavity Steiner insertion-attempt limit exceeded",
            ));
        }
        self.check_cancelled(constraint_index, self.cavity_steiner_insertion_attempts)
    }

    fn check_cancelled(
        &self,
        constraint_index: u32,
        step: u64,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        if step.is_multiple_of(
            self.options
                .segment_recovery
                .constraints
                .cancellation_check_interval,
        ) {
            self.poll_cancellation(constraint_index)?;
        }
        Ok(())
    }

    #[track_caller]
    fn poll_cancellation(&self, constraint_index: u32) -> Result<(), DelaunayFacetRecoveryError> {
        if self.cancellation.is_cancelled() {
            return Err(error(
                DelaunayFacetRecoveryErrorKind::Cancelled,
                Some(constraint_index),
                "cancelled",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::NeverCancelled;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    /// The battery cold plate's boundary hands facet recovery 1,688,558 facets. Each needed
    /// about 89 support steps (606 at most), and one account shared by all of them ran out of
    /// its 100,000,000 steps at facet 1,126,119 with no defect there. Each facet now has its own
    /// account: any number of facets within the bound recover, and one facet past it still fails
    /// at that facet.
    #[test]
    fn each_facet_has_its_own_work_account() {
        let options = DelaunayFacetRecoveryOptions {
            maximum_search_steps: 4,
            maximum_flip_attempts: 4,
            maximum_support_steps: 4,
            maximum_cavity_steps: 4,
            ..DelaunayFacetRecoveryOptions::default()
        };
        let mut work = FacetRecoveryWork::new(options, &NeverCancelled);
        for constraint in 0..1_000 {
            work.begin_constraint(constraint).unwrap();
            for _ in 0..4 {
                work.support_step(constraint).unwrap();
                work.search_step(constraint).unwrap();
                work.flip_attempt(constraint).unwrap();
                work.cavity_step(constraint).unwrap();
            }
        }
        work.begin_constraint(1_000).unwrap();
        for _ in 0..4 {
            work.support_step(1_000).unwrap();
        }
        let error = work.support_step(1_000).unwrap_err();
        assert_eq!(error.kind, DelaunayFacetRecoveryErrorKind::ResourceLimit);
        assert_eq!(error.constraint_index, Some(1_000));
    }

    struct CancellationFlag(AtomicBool);

    impl MeshingCancellationSignal for CancellationFlag {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::Relaxed)
        }
    }

    #[test]
    fn short_facets_keep_a_cancellation_checkpoint_when_accounts_restart() {
        let cancellation = CancellationFlag(AtomicBool::new(false));
        let mut work = FacetRecoveryWork::new(Default::default(), &cancellation);
        for constraint in 0..1_000 {
            work.begin_constraint(constraint).unwrap();
            work.support_step(constraint).unwrap();
            work.search_step(constraint).unwrap();
        }
        cancellation.0.store(true, Ordering::Relaxed);
        let error = work.begin_constraint(1_000).unwrap_err();
        assert_eq!(error.kind, DelaunayFacetRecoveryErrorKind::Cancelled);
        assert_eq!(error.constraint_index, Some(1_000));
    }

    #[test]
    fn cancellation_inside_a_facet_retains_its_work_interval() {
        let cancellation = CancellationFlag(AtomicBool::new(false));
        let mut options = DelaunayFacetRecoveryOptions::default();
        options
            .segment_recovery
            .constraints
            .cancellation_check_interval = 2;
        let mut work = FacetRecoveryWork::new(options, &cancellation);
        work.begin_constraint(7).unwrap();
        cancellation.0.store(true, Ordering::Relaxed);
        work.support_step(7).unwrap();
        let error = work.support_step(7).unwrap_err();
        assert_eq!(error.kind, DelaunayFacetRecoveryErrorKind::Cancelled);
        assert_eq!(error.constraint_index, Some(7));
    }
}
