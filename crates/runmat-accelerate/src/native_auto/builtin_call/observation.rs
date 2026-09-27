use super::*;
use crate::placement::LocalPlacementFeedback;
use runmat_accelerate_api::observation::{
    ProviderExecutionDisposition, ProviderObservationSupport, ProviderOperationReceipt,
    ProviderOperationRecorder,
};

#[derive(Default)]
pub(in crate::native_auto) struct BuiltinPlanningContext<'a> {
    pub(in crate::native_auto) runtime: Option<&'a RuntimeContext>,
    pub(in crate::native_auto) preparation_program:
        Option<runmat_accelerate_api::ProviderPreparationProgram<'a>>,
}

#[derive(Default)]
pub(in crate::native_auto) struct BuiltinObservationPlan {
    pub(in crate::native_auto) feedback: Option<Box<LocalPlacementFeedback>>,
    pub(in crate::native_auto) provider_query:
        Option<Box<runmat_accelerate_api::ProviderFeasibilityQuery>>,
    pub(in crate::native_auto) provider_observation: bool,
}

pub struct BuiltinCallObservation {
    pub(super) feedback: Option<Box<LocalPlacementFeedback>>,
    pub(super) trace: Option<PlacementTraceContext>,
    provider_selected: bool,
    host_inputs: bool,
    started: Instant,
    receipt: Option<ProviderOperationReceipt>,
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
            receipt: None,
        }
    }

    pub(super) fn prepare_provider(
        &mut self,
        provider: &dyn AccelProvider,
        query: Option<&runmat_accelerate_api::ProviderFeasibilityQuery>,
    ) {
        if !self.provider_selected || self.feedback.is_none() {
            return;
        }
        if let Some(query) = query {
            if let ProviderObservationSupport::Observed(mut receipt) =
                provider.begin_operation_observation(query)
            {
                receipt.require_provider_execution(provider.device_id(), query.operation.clone());
                self.receipt = Some(receipt);
            }
        }
    }

    pub(super) fn recorder(&self) -> Option<ProviderOperationRecorder> {
        self.receipt
            .as_ref()
            .map(ProviderOperationReceipt::recorder)
    }

    pub(super) fn execution_started(&mut self) {
        // The host candidate estimates builtin execution, not argument staging.
        // Upload timing remains a separate trace event.
        self.started = Instant::now();
    }

    pub(super) fn preparation_fallback(&mut self) {
        if let Some(recorder) = self.recorder() {
            recorder.execution_disposition(ProviderExecutionDisposition::SharedRuntimeFallback);
        }
        self.receipt = None;
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

    pub(super) fn cleanup_failed(&mut self) {
        if let Some(receipt) = self.receipt.as_mut() {
            receipt.fail(
                runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
            );
        }
        self.report(false);
        if let Some(trace) = self.trace.take() {
            trace.fallback(None, "provider_cleanup_failed", None);
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
            if let (Some(receipt), Some(feedback)) = (self.receipt.take(), self.feedback.take()) {
                // The receipt independently requires explicit operation-owner
                // route evidence. Resident/host result shape proves nothing.
                crate::placement::publish_provider_feedback(receipt, *feedback);
            }
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
