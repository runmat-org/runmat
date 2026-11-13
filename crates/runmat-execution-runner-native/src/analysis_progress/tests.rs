use super::*;

#[test]
fn local_context_shares_cancellation_and_restores_the_callers_context() {
    let previous = FeaCancellationToken::new();
    previous.cancel();
    let _previous = runmat_analysis_fea::replace_fea_progress_context(FeaProgressContext {
        cancellation: previous,
        ..Default::default()
    });
    let current = FeaCancellationToken::new();
    {
        let _local = install(current.clone(), None);
        assert!(!runmat_analysis_fea::is_fea_cancelled());
        current.cancel();
        assert!(runmat_analysis_fea::is_fea_cancelled());
    }
    assert!(runmat_analysis_fea::is_fea_cancelled());
}

#[derive(Clone, Default)]
struct RecordedEvents(Arc<std::sync::Mutex<Vec<String>>>);

impl tracing::Subscriber for RecordedEvents {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Fields(String);
        impl tracing::field::Visit for Fields {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                use std::fmt::Write;
                write!(self.0, "{}={value:?};", field.name()).unwrap();
            }
        }
        let mut fields = Fields(String::new());
        event.record(&mut fields);
        self.0.lock().unwrap().push(fields.0);
    }
}

#[test]
fn local_and_remote_progress_reach_tracing_without_losing_the_solver_message() {
    let recorded = RecordedEvents::default();
    let remote = AttemptId::derive(&[b"progress-transport-test"]);
    tracing::subscriber::with_default(recorded.clone(), || {
        for attempt in [None, Some(remote)] {
            let _context = install(FeaCancellationToken::new(), attempt);
            runmat_analysis_fea::progress::emit_phase(
                "Conjugate gradient",
                runmat_analysis_fea::progress::FeaProgressPhase::Solve,
                runmat_analysis_fea::progress::FeaProgressStatus::Advanced,
                "iteration=7 recursive_relative_residual=1.0e-9",
                Some(7),
                None,
            );
        }
    });
    let events = recorded.0.lock().unwrap();
    assert_eq!(events.len(), 2);
    for event in events.iter() {
        assert!(event.contains("analysis.solver.progress"));
        assert!(event.contains("Conjugate gradient"));
        assert!(event.contains("iteration=7 recursive_relative_residual=1.0e-9"));
        assert!(event.contains("current=Some(7)"));
    }
    assert!(events[0].contains("attempt_id=\"\""));
    assert!(events[1].contains(&remote.to_string()));
}
