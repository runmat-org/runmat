use crate::placement::LocalPlacementFeedback;
use runmat_time::{duration_ns_saturating, Instant};

/// A placement choice is not an execution failure. The caller runs the shared
/// bytecode path and retains this observation until that region actually exits.
pub enum FusionExecutionOutcome<T> {
    /// Provider dispatch produced a value/handle. This does not establish
    /// completion of asynchronous device work or provide execution timing.
    Executed(T),
    SharedRuntime(FusionCpuObservation),
}

/// Measures completion of a shared-runtime bytecode region on the host.
/// Inner calls may submit provider work; this elapsed time is not evidence of
/// completion of that device work and must not be assigned to a GPU candidate.
pub struct FusionCpuObservation {
    feedback: Option<Box<LocalPlacementFeedback>>,
    started: Instant,
}

impl FusionCpuObservation {
    pub(super) fn new(feedback: Option<Box<LocalPlacementFeedback>>) -> Self {
        Self {
            feedback,
            started: Instant::now(),
        }
    }

    /// Only the owner of the complete bytecode region may report success.
    /// Success means host bytecode exited, not that nested GPU queues drained.
    pub fn complete(mut self) {
        self.observe(true);
    }

    fn observe(&mut self, succeeded: bool) {
        let Some(feedback) = self.feedback.take() else {
            return;
        };
        if let Some(service) = feedback.runtime.service_ports().placement() {
            let _ = service.observe(runmat_execution::PlacementFeedback {
                signature: feedback.signature,
                candidate: feedback.candidate,
                total_elapsed_ns: duration_ns_saturating(self.started.elapsed()),
                succeeded,
            });
        }
    }
}

impl Drop for FusionCpuObservation {
    fn drop(&mut self) {
        self.observe(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion_exec::FusionPlacementSelection;
    use crate::placement::{FusionPlacementObserver, PlacementSession};
    use runmat_execution::{Digest, PlacementRevision, PlacementSignature};
    use runmat_runtime::context::{RuntimeContext, RuntimeServicePorts};
    use std::rc::Rc;

    fn ticket(candidate: &str) -> (LocalPlacementFeedback, Rc<PlacementSession>) {
        let session = Rc::new(PlacementSession::default());
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_placement(session.clone()));
        let digest = || Digest::sha256(b"fusion-observation-test");
        (
            LocalPlacementFeedback {
                runtime,
                signature: PlacementSignature {
                    region: None,
                    operation: "fusion.observation.test".into(),
                    runtime_facts: digest(),
                    revision: PlacementRevision {
                        program: None,
                        catalog: digest(),
                        compiler: digest(),
                        provider: digest(),
                        policy: digest(),
                    },
                },
                candidate: candidate.into(),
            },
            session,
        )
    }

    #[test]
    fn cpu_observation_records_success_once_only_after_completion() {
        let (feedback, session) = ticket("cpu.shared.test");
        let observation = FusionCpuObservation::new(Some(Box::new(feedback)));
        assert!(session.profile_snapshot().feedback.is_empty());
        observation.complete();
        let snapshot = session.profile_snapshot();
        assert_eq!(snapshot.feedback.len(), 1);
        assert_eq!(snapshot.feedback[0].samples, 1);
        assert_eq!(snapshot.feedback[0].failures, 0);
    }

    #[test]
    fn abandoned_cpu_observation_never_records_success() {
        let (feedback, session) = ticket("cpu.shared.test");
        drop(FusionCpuObservation::new(Some(Box::new(feedback))));
        let snapshot = session.profile_snapshot();
        assert_eq!(snapshot.feedback[0].samples, 0);
        assert_eq!(snapshot.feedback[0].failures, 1);
    }

    #[test]
    fn cancelled_future_drops_cpu_ticket_without_a_success_sample() {
        let (feedback, session) = ticket("cpu.shared.test");
        let observation = FusionCpuObservation::new(Some(Box::new(feedback)));
        let future = async move {
            std::future::pending::<()>().await;
            observation.complete();
        };
        drop(future);
        let snapshot = session.profile_snapshot();
        assert_eq!(snapshot.feedback[0].samples, 0);
        assert_eq!(snapshot.feedback[0].failures, 1);
    }

    #[test]
    fn gpu_submission_is_not_completed_execution_feedback() {
        let (feedback, session) = ticket("gpu.fusion.test");
        FusionPlacementSelection {
            observer: FusionPlacementObserver::new(None),
            feedback: Some(feedback),
            started: Instant::now(),
        }
        .submitted();
        assert!(session.profile_snapshot().feedback.is_empty());
    }

    #[test]
    fn failed_gpu_submission_retains_failure_feedback() {
        let (feedback, session) = ticket("gpu.fusion.test");
        drop(FusionPlacementSelection {
            observer: FusionPlacementObserver::new(None),
            feedback: Some(feedback),
            started: Instant::now(),
        });
        let snapshot = session.profile_snapshot();
        assert_eq!(snapshot.feedback[0].samples, 0);
        assert_eq!(snapshot.feedback[0].failures, 1);
    }
}
