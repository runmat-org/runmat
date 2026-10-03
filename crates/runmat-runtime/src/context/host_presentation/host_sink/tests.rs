use super::*;
use crate::context::host_presentation::tests::Witness;
use crate::context::RuntimeWholeWorkSelection;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::{
    CancellationReason, PlacementRevision, WholeWorkDecision, WholeWorkDecisionKind,
};
use std::rc::Rc;

fn selected() -> RuntimeWholeWorkSelection {
    let digest = Digest::sha256(b"host-sink-cancellation");
    RuntimeWholeWorkSelection {
        decision: WholeWorkDecision {
            signature: PlacementSignature {
                region: None,
                operation: "gray2rgb".into(),
                runtime_facts: digest,
                revision: PlacementRevision {
                    program: None,
                    catalog: digest,
                    compiler: digest,
                    provider: digest,
                    policy: digest,
                },
            },
            selected: WholeWorkCandidate {
                kind: ExecutionCandidateKind::SharedRuntime,
                execution_location: CandidateExecutionLocation::Host,
                preparation: CandidatePreparationState::Ready,
                output_residency: CandidateOutputResidency::Host,
                presentation: CandidateOutputPresentation::Direct,
                route_facts: digest,
                cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
                resources: CandidateResourceDemand::default(),
                transactional_results: true,
            }
            .selected("gray2rgb"),
            kind: WholeWorkDecisionKind::Unmeasured,
            predicted_total_ns: None,
        },
        probe: None,
    }
}

#[test]
fn cancellation_and_abandoned_host_sink_never_publish_feedback() {
    let witness = Rc::new(Witness::default());
    let cancellation = CancellationSource::new();
    let runtime = witness
        .context()
        .with_cancellation_token(cancellation.token());
    drop(HostSinkReceipt::new(runtime.clone(), selected()));
    assert!(witness.feedback().is_empty());
    cancellation.cancel(CancellationReason::User);
    HostSinkReceipt::new(runtime, selected()).finish(true);
    assert!(witness.feedback().is_empty());
}
