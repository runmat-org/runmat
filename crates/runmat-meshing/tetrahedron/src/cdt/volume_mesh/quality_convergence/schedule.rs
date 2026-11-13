//! Coupled size/shape scheduling decisions, independent of mesh types.
//! A repair rejection is provisional: it describes one unchanged state and is
//! discarded by any accepted change. It becomes terminal only when refinement
//! cannot unlock it.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RefinementStep {
    /// No refinement violation remained, so refinement was not attempted.
    Idle,
    Accepted,
    /// A bounded source window failed while shape violations remain.
    Yielded,
    /// A complete pass admitted nothing.
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RepairStep {
    Changed,
    Unchanged,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Next<F> {
    Continue,
    Finish,
    /// Global exhaustion: the retained repair rejection is terminal.
    Reject(F),
    /// Neither strategy changed state and repair reported no rejection.
    NoProgress,
}

pub(super) struct CoupledSchedule<F> {
    blocked: Option<F>,
}

impl<F> Default for CoupledSchedule<F> {
    fn default() -> Self {
        Self { blocked: None }
    }
}

impl<F> CoupledSchedule<F> {
    pub(super) fn refined(&mut self, step: RefinementStep) {
        if step == RefinementStep::Accepted {
            self.blocked = None;
        }
    }

    /// An unchanged state already rejected by repair is never re-searched. While refinement
    /// admits batches, shape repair waits: new points keep replacing cells, so repairing the
    /// shape defects of every intermediate state re-searches slivers refinement is about to
    /// remove. Repair runs once refinement yields to a shape defect, is blocked, or is done.
    pub(super) fn should_repair(&self, step: RefinementStep, shape_violation: bool) -> bool {
        self.blocked.is_none()
            && step != RefinementStep::Accepted
            && (step == RefinementStep::Blocked || shape_violation)
    }

    pub(super) fn repaired(&mut self, step: RepairStep, rejection: Option<F>) {
        debug_assert!(step == RepairStep::Unchanged || rejection.is_none());
        self.blocked = rejection;
    }

    pub(super) fn next(
        &mut self,
        refinement: RefinementStep,
        repair: Option<RepairStep>,
        converged: bool,
    ) -> Next<F> {
        if refinement == RefinementStep::Blocked {
            if let Some(failure) = self.blocked.take() {
                return Next::Reject(failure);
            }
        }
        if converged {
            return Next::Finish;
        }
        if refinement == RefinementStep::Yielded {
            return Next::Continue;
        }
        if refinement == RefinementStep::Accepted || repair == Some(RepairStep::Changed) {
            return Next::Continue;
        }
        match self.blocked.take() {
            Some(failure) => Next::Reject(failure),
            None => Next::NoProgress,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RefinementStep::*, RepairStep::*, *};

    // witness: FEA-MESH-DISCRETIZATION-001/volume_quality_convergence @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
    //   boundary: A failed source window yields to repair and a rejected repair resumes refinement;
    //   fairness on real parts is not asserted.
    #[test]
    fn failed_source_window_yields_and_rejected_repair_resumes_refinement() {
        let mut schedule = CoupledSchedule::default();
        schedule.refined(Yielded);
        assert!(schedule.should_repair(Yielded, true));
        schedule.repaired(Unchanged, Some("no admissible move"));
        // Yield is not exhaustion: refinement gets its complete pass.
        assert_eq!(
            schedule.next(Yielded, Some(Unchanged), false),
            Next::Continue
        );
        // The unchanged rejected state is not searched again.
        assert!(!schedule.should_repair(Yielded, true));
        schedule.refined(Accepted);
        // Repair waits while refinement admits batches.
        assert!(!schedule.should_repair(Accepted, true));
        assert_eq!(schedule.next(Accepted, None, false), Next::Continue);
        // Accepted refinement made the rejection stale; once refinement is done, repair runs.
        schedule.refined(Idle);
        assert!(schedule.should_repair(Idle, true));
        schedule.repaired(Changed, None);
        assert_eq!(schedule.next(Idle, Some(Changed), true), Next::Finish);
    }

    #[test]
    fn rejection_is_terminal_only_when_refinement_is_also_exhausted() {
        let mut schedule = CoupledSchedule::default();
        schedule.refined(Yielded);
        schedule.repaired(Unchanged, Some("rejected"));
        assert_eq!(
            schedule.next(Yielded, Some(Unchanged), false),
            Next::Continue
        );
        schedule.refined(Blocked);
        assert!(!schedule.should_repair(Blocked, true));
        assert_eq!(
            schedule.next(Blocked, None, false),
            Next::Reject("rejected")
        );
    }

    #[test]
    fn shape_only_rejection_with_idle_refinement_is_terminal() {
        let mut schedule = CoupledSchedule::default();
        schedule.refined(Idle);
        assert!(schedule.should_repair(Idle, true));
        schedule.repaired(Unchanged, Some("rejected"));
        assert_eq!(
            schedule.next(Idle, Some(Unchanged), false),
            Next::Reject("rejected")
        );
    }

    #[test]
    fn unchanged_repair_without_rejection_is_no_progress_not_a_loop() {
        let mut schedule = CoupledSchedule::<&str>::default();
        schedule.refined(Idle);
        schedule.repaired(Unchanged, None);
        assert_eq!(
            schedule.next(Idle, Some(Unchanged), false),
            Next::NoProgress
        );
    }

    #[test]
    fn blocked_refinement_is_repaired_before_any_verdict() {
        let mut schedule = CoupledSchedule::<&str>::default();
        schedule.refined(Blocked);
        assert!(schedule.should_repair(Blocked, false));
        schedule.repaired(Changed, None);
        assert_eq!(schedule.next(Blocked, Some(Changed), false), Next::Continue);
        assert!(!schedule.should_repair(Accepted, false));
    }

    #[test]
    fn convergence_wins_over_an_unrelated_progress_check() {
        let mut schedule = CoupledSchedule::<&str>::default();
        schedule.refined(Idle);
        assert!(!schedule.should_repair(Idle, false));
        assert_eq!(schedule.next(Idle, None, true), Next::Finish);
    }

    /// Scripted strategies: every bounded script terminates with a verdict
    /// and never retries repair on a state it has already rejected.
    #[test]
    fn all_short_outcome_scripts_terminate_without_redundant_repair() {
        let refinements = [Idle, Accepted, Yielded, Blocked];
        let repairs = [(Changed, false), (Unchanged, false), (Unchanged, true)];
        let choices = refinements
            .iter()
            .flat_map(|r| repairs.iter().map(move |p| (*r, *p)))
            .collect::<Vec<_>>();
        const LENGTH: u32 = 4;
        let scripts = (0..choices.len().pow(LENGTH)).map(|mut code| {
            (0..LENGTH)
                .map(|_| {
                    let choice = choices[code % choices.len()];
                    code /= choices.len();
                    choice
                })
                .collect::<Vec<_>>()
        });
        for script in scripts {
            let mut schedule = CoupledSchedule::default();
            let mut rejected_state = false;
            for (iteration, (refinement, (repair, rejects))) in script.iter().enumerate() {
                schedule.refined(*refinement);
                if *refinement == Accepted {
                    rejected_state = false;
                }
                let ran = schedule.should_repair(*refinement, true).then(|| {
                    assert!(
                        !rejected_state,
                        "repair retried an unchanged rejected state"
                    );
                    let rejection = (*repair == Unchanged && *rejects).then_some(iteration);
                    rejected_state = rejection.is_some();
                    schedule.repaired(*repair, rejection);
                    *repair
                });
                match schedule.next(*refinement, ran, false) {
                    Next::Continue => {}
                    Next::Reject(_) | Next::NoProgress => break,
                    Next::Finish => unreachable!(),
                }
            }
        }
    }
}
