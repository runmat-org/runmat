use super::*;
use crate::placement::LocalPlacementFeedback;

pub struct BuiltinCallObservation {
    pub(super) feedback: Option<Box<LocalPlacementFeedback>>,
    pub(super) trace: Option<PlacementTraceContext>,
    provider_selected: bool,
    host_inputs: bool,
    started: Instant,
}

impl BuiltinCallObservation {
    pub(super) fn new(
        feedback: Option<Box<LocalPlacementFeedback>>,
        trace: PlacementTraceContext,
        provider_selected: bool,
        args: &[Value],
    ) -> Self {
        Self {
            feedback,
            trace: Some(trace),
            provider_selected,
            host_inputs: !args
                .iter()
                .any(runmat_runtime::dispatcher::value_contains_gpu),
            started: Instant::now(),
        }
    }

    pub(super) fn execution_started(&mut self) {
        // The host candidate estimates builtin execution, not argument staging.
        // Upload timing remains a separate trace event.
        self.started = Instant::now();
    }

    pub(super) fn preparation_fallback(&mut self) {
        self.report(false);
        self.provider_selected = false;
        if let Some(trace) = self.trace.as_ref() {
            trace.event(
                PlacementEventKind::Fallback,
                Some(PlacementVariant::SharedRuntime),
                Some("provider_preparation_retained_host"),
                None,
                None,
                &[],
            );
        }
    }

    pub(super) fn preparation_only(mut self) {
        self.feedback = None;
        if let Some(trace) = self.trace.take() {
            trace.fallback(None, "arguments_prepared_without_execution", None);
        }
    }

    pub fn finish(mut self, result: &Result<Value, RuntimeError>) {
        let Ok(value) = result else { return };
        if self
            .feedback
            .as_ref()
            .is_some_and(|ticket| ticket.runtime.is_cancelled())
        {
            return;
        }
        let elapsed = duration_ns_saturating(self.started.elapsed()).max(1);
        if self.provider_selected {
            // A returned value may follow asynchronous submission or the
            // dispatcher's host retry. Neither proves provider completion.
            self.feedback = None;
            if let Some(trace) = self.trace.take() {
                if runmat_runtime::dispatcher::value_contains_gpu(value) {
                    trace.submitted(PlacementVariant::ProviderOperation, Some(elapsed));
                } else {
                    trace.fallback(
                        Some(PlacementVariant::SharedRuntime),
                        "provider_call_returned_host",
                        Some(elapsed),
                    );
                }
            }
        } else if self.host_inputs && !runmat_runtime::dispatcher::value_contains_gpu(value) {
            self.report(true);
            if let Some(trace) = self.trace.take() {
                trace.complete(Some(PlacementVariant::SharedRuntime), Some(elapsed));
            }
        } else {
            self.feedback = None;
            if let Some(trace) = self.trace.take() {
                trace.fallback(None, "execution_completion_unproven", Some(elapsed));
            }
        }
    }

    fn report(&mut self, succeeded: bool) {
        if let Some(ticket) = self.feedback.take() {
            if let Some(service) = ticket.runtime.service_ports().placement() {
                let _ = service.observe(runmat_execution::PlacementFeedback {
                    signature: ticket.signature,
                    candidate: ticket.candidate,
                    total_elapsed_ns: duration_ns_saturating(self.started.elapsed()).max(1),
                    succeeded,
                });
            }
        }
    }
}

impl Drop for BuiltinCallObservation {
    fn drop(&mut self) {
        self.report(false);
    }
}
