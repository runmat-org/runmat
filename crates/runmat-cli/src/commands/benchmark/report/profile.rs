use runmat_accelerate::placement::{PlacementFeedbackSeries, PlacementProfileSnapshot};
use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum History {
    ObservedDelta,
    NoObservedFeedback,
    EvictedOrReset,
    CounterResetOrOverflow,
}

#[derive(Serialize)]
pub(super) struct Delta<'a> {
    pub history: History,
    /// Completed feedback only, not every selected/declined execution route.
    pub series: Vec<Series<'a>>,
}

#[derive(Serialize)]
pub(super) struct Series<'a> {
    pub signature: &'a runmat_execution::PlacementSignature,
    pub candidate: &'a str,
    pub kind: runmat_execution::ExecutionCandidateKind,
    pub samples: u64,
    pub failures: u64,
    pub total_elapsed_ns: String,
}

fn same_key(a: &PlacementFeedbackSeries, b: &PlacementFeedbackSeries) -> bool {
    a.signature == b.signature && a.candidate == b.candidate && a.kind == b.kind
}

pub(super) fn delta<'a>(
    before: &PlacementProfileSnapshot,
    after: &'a PlacementProfileSnapshot,
) -> Delta<'a> {
    let lost = before.schema_version != after.schema_version
        || before
            .feedback
            .iter()
            .any(|old| !after.feedback.iter().any(|new| same_key(old, new)));
    let mut history = if lost {
        History::EvictedOrReset
    } else {
        History::NoObservedFeedback
    };
    let mut series = Vec::with_capacity(after.feedback.len());
    for new in &after.feedback {
        let old = before.feedback.iter().find(|old| same_key(old, new));
        let (samples, failures, elapsed) = match old {
            Some(old) => (
                new.samples.checked_sub(old.samples),
                new.failures.checked_sub(old.failures),
                new.total_elapsed_ns.checked_sub(old.total_elapsed_ns),
            ),
            None => (
                Some(new.samples),
                Some(new.failures),
                Some(new.total_elapsed_ns),
            ),
        };
        let (Some(samples), Some(failures), Some(elapsed)) = (samples, failures, elapsed) else {
            history = History::CounterResetOrOverflow;
            continue;
        };
        if samples == 0 && failures == 0 {
            continue;
        }
        if matches!(history, History::NoObservedFeedback) {
            history = History::ObservedDelta;
        }
        series.push(Series {
            signature: &new.signature,
            candidate: &new.candidate,
            kind: new.kind,
            samples,
            failures,
            total_elapsed_ns: elapsed.to_string(),
        });
    }
    Delta { history, series }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn profile_deltas_do_not_claim_unobserved_or_reset_routes() {
        let empty = PlacementProfileSnapshot {
            schema_version: 1,
            feedback: Vec::new(),
        };
        assert!(matches!(
            delta(&empty, &empty).history,
            History::NoObservedFeedback
        ));
        let changed = PlacementProfileSnapshot {
            schema_version: 2,
            feedback: Vec::new(),
        };
        assert!(matches!(
            delta(&empty, &changed).history,
            History::EvictedOrReset
        ));
        let digest = runmat_execution::Digest::sha256(b"profile test");
        let series = PlacementFeedbackSeries {
            signature: runmat_execution::PlacementSignature {
                region: None,
                operation: "test".into(),
                runtime_facts: digest,
                provider_provenance: None,
                graph_output_authority: None,
                revision: runmat_execution::PlacementRevision {
                    program: None,
                    catalog: digest,
                    compiler: digest,
                    policy: digest,
                    authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(
                        digest,
                    ),
                },
            },
            candidate: "canonical-test-candidate".into(),
            kind: runmat_execution::ExecutionCandidateKind::SharedRuntime,
            samples: 2,
            failures: 0,
            total_elapsed_ns: 20,
            total_squared_elapsed_ns: 200,
            latest_tick: 2,
        };
        let before = PlacementProfileSnapshot {
            schema_version: 1,
            feedback: vec![series],
        };
        assert!(matches!(
            delta(&before, &empty).history,
            History::EvictedOrReset
        ));
        let mut after = before.clone();
        after.feedback[0].samples = 1;
        assert!(matches!(
            delta(&before, &after).history,
            History::CounterResetOrOverflow
        ));
        assert!(delta(&before, &after).series.is_empty());
        after.feedback[0].samples = 3;
        after.feedback[0].total_elapsed_ns = 35;
        let observed = delta(&before, &after);
        assert!(matches!(observed.history, History::ObservedDelta));
        assert_eq!(observed.series[0].samples, 1);
        assert_eq!(observed.series[0].total_elapsed_ns, "15");
    }
}
