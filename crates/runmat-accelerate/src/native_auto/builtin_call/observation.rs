use super::*;
use crate::placement::LocalPlacementFeedback;
use runmat_accelerate_api::observation::{
    ProviderExecutionDisposition, ProviderObservationSupport, ProviderOperationReceipt,
    ProviderOperationRecorder,
};
use runmat_execution::profiling::{CommandProviderContribution, ProviderObservationTerminal};

#[derive(Default)]
pub(in crate::native_auto) struct BuiltinPlanningContext<'a> {
    pub(in crate::native_auto) compound:
        Option<&'a runmat_runtime::context::preparation::AdmittedBuiltinCompound>,
    pub(in crate::native_auto) runtime: Option<&'a RuntimeContext>,
    pub(in crate::native_auto) preparation_program:
        Option<runmat_accelerate_api::ProviderPreparationProgram<'a>>,
    pub(in crate::native_auto) prepared_authority:
        Option<runmat_runtime::context::preparation::PreparedBindingAuthority>,
    pub(in crate::native_auto) source_decline: crate::placement::LocalSourceDecline,
    pub(in crate::native_auto) host: Option<(
        &'a crate::fusion_exec::host_readback::CpuHostResources,
        runmat_execution::Digest,
    )>,
}

#[derive(Default)]
pub(in crate::native_auto) struct BuiltinObservationPlan {
    pub(in crate::native_auto) feedback: Option<Box<LocalPlacementFeedback>>,
    pub(in crate::native_auto) provider_query:
        Option<Box<runmat_accelerate_api::ProviderFeasibilityQuery>>,
    pub(in crate::native_auto) provider_observation: bool,
    pub(in crate::native_auto) source_declined: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PreparationRoute {
    Host,
    Provider,
    // Provider preparation did not produce an executable provider route.
    // This is not a new host selection and carries no completion right.
    SharedFallback,
}

pub struct BuiltinCallObservation {
    pub(super) feedback: Option<Box<LocalPlacementFeedback>>,
    pub(super) trace: Option<PlacementTraceContext>,
    route: PreparationRoute,
    host_inputs: bool,
    started: Instant,
    receipt: Option<ProviderOperationReceipt>,
    command_contribution: Option<CommandProviderContribution>,
    completion: Option<runmat_runtime::context::preparation::PreparedCompletionRight>,
    host_execution: Option<runmat_runtime::context::preparation::PreparedHostExecution>,
}

struct BackendCompletedCall {
    binding: runmat_builtins::BuiltinBindingIdentity,
    feedback: Box<LocalPlacementFeedback>,
    receipt: Option<ProviderOperationReceipt>,
    command_contribution: Option<CommandProviderContribution>,
    status: runmat_runtime::context::preparation::PreparedCallStatus,
}

impl runmat_runtime::context::preparation::PreparedCallCompletion for BackendCompletedCall {
    fn binding(&self) -> runmat_builtins::BuiltinBindingIdentity {
        self.binding
    }

    fn status(&self) -> runmat_runtime::context::preparation::PreparedCallStatus {
        self.status
    }

    fn publish(self: Box<Self>, outer_started: Instant, elapsed_ns: u64) {
        use runmat_runtime::context::preparation::PreparedCallStatus;
        let Self {
            feedback,
            receipt,
            command_contribution,
            status,
            ..
        } = *self;
        if feedback.runtime.is_cancelled() {
            return;
        }
        if status == PreparedCallStatus::CompletedFailure {
            if let Some(contribution) = command_contribution {
                contribution.complete(ProviderObservationTerminal::Failed);
            }
            feedback.observe(elapsed_ns, false);
            return;
        }
        if status == PreparedCallStatus::Succeeded {
            if let Some(receipt) = receipt {
                // Only receipts with an explicitly shared monotonic host clock
                // can combine the pre-receipt call prefix with provider proof.
                let Some(prelude) = receipt.default_clock_prelude_since(outer_started) else {
                    return;
                };
                crate::placement::publish_provider_feedback_full_call(
                    receipt,
                    *feedback,
                    prelude,
                    command_contribution,
                );
                return;
            }
        }
        feedback.observe(elapsed_ns, status == PreparedCallStatus::Succeeded);
    }
}

impl BuiltinCallObservation {
    pub(super) fn select_private_host(
        &mut self,
        context: &RuntimeContext,
        form: &'static runmat_builtins::BuiltinPreparationForm,
    ) -> Option<runmat_runtime::context::preparation::PreparedHostExecution> {
        if !self.host_selected() {
            return None;
        }
        let proof = self
            .completion
            .as_ref()?
            .select_host_execution(context, form)?;
        self.host_execution = Some(proof.clone());
        Some(proof)
    }

    pub(super) fn host_selected(&self) -> bool {
        self.route == PreparationRoute::Host
    }

    pub(super) fn provider_selected(&self) -> bool {
        self.route == PreparationRoute::Provider
    }

    #[cfg(test)]
    pub(super) fn shared_fallback_selected(&self) -> bool {
        self.route == PreparationRoute::SharedFallback
    }
    pub(super) fn new(
        feedback: Option<Box<LocalPlacementFeedback>>,
        trace: PlacementTraceContext,
        provider_selected: bool,
        args: &[Value],
    ) -> Self {
        Self {
            feedback,
            trace: Some(trace),
            route: if provider_selected {
                PreparationRoute::Provider
            } else {
                PreparationRoute::Host
            },
            host_inputs: !args
                .iter()
                .any(runmat_runtime::dispatcher::value_contains_gpu),
            started: Instant::now(),
            receipt: None,
            command_contribution: None,
            completion: None,
            host_execution: None,
        }
    }

    pub(super) fn with_completion(
        mut self,
        completion: Option<runmat_runtime::context::preparation::PreparedCompletionRight>,
    ) -> Self {
        self.completion = completion;
        self
    }

    fn deposit_failure(&mut self) -> bool {
        let Some(right) = self.completion.take() else {
            return false;
        };
        if let Some(feedback) = self.feedback.take() {
            let binding = right.binding();
            right.deposit(Box::new(BackendCompletedCall {
                binding,
                feedback,
                receipt: None,
                command_contribution: self.command_contribution.take(),
                status: runmat_runtime::context::preparation::PreparedCallStatus::CompletedFailure,
            }));
        }
        true
    }

    pub(super) fn prepare_provider(
        &mut self,
        provider: &dyn AccelProvider,
        query: Option<&runmat_accelerate_api::ProviderFeasibilityQuery>,
    ) {
        if !self.provider_selected()
            || !self
                .feedback
                .as_ref()
                .is_some_and(|feedback| feedback.is_current())
        {
            return;
        }
        if let Some(query) = query {
            if let ProviderObservationSupport::Observed(mut receipt) =
                provider.begin_operation_observation(query)
            {
                receipt.require_provider_execution(provider.device_id(), query.operation.clone());
                self.command_contribution = self
                    .feedback
                    .as_ref()
                    .and_then(|feedback| feedback.runtime.command_profiling())
                    .and_then(|command| command.begin_provider_operation());
                self.receipt = Some(receipt);
            }
        }
    }

    pub(super) fn recorder(&self) -> Option<ProviderOperationRecorder> {
        self.receipt
            .as_ref()
            .map(ProviderOperationReceipt::recorder)
    }
    pub(super) fn prepare_compound(
        &mut self,
        provider: &dyn AccelProvider,
        query: Option<&runmat_accelerate_api::ProviderFeasibilityQuery>,
        plan: &runmat_runtime::context::preparation::AdmittedBuiltinCompound,
    ) {
        if !self.provider_selected()
            || !self
                .feedback
                .as_ref()
                .is_some_and(|feedback| feedback.is_current())
        {
            return;
        }
        let Some(query) = query else {
            return;
        };
        if let ProviderObservationSupport::Observed(mut receipt) = plan.begin_observation(provider)
        {
            plan.require_execution(&mut receipt, query.operation.clone());
            self.command_contribution = self
                .feedback
                .as_ref()
                .and_then(|feedback| feedback.runtime.command_profiling())
                .and_then(|command| command.begin_provider_operation());
            self.receipt = Some(receipt);
        }
    }

    pub(super) fn preparation_fallback(&mut self) {
        if let Some(recorder) = self.recorder() {
            recorder.execution_disposition(ProviderExecutionDisposition::SharedRuntimeFallback);
        }
        self.receipt = None;
        self.command_contribution = None;
        if self.completion.take().is_some() {
            // Selected provider preparation did not become the promised route.
            // Do not treat its host fallback as a completed v3 candidate.
            self.feedback = None;
        }
        self.report(false);
        self.route = PreparationRoute::SharedFallback;
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

    pub(super) fn preparation_failed(&mut self) {
        let reason = if self.cancelled() {
            runmat_accelerate_api::observation::ProviderObservationFailure::Cancelled
        } else {
            runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed
        };
        if let Some(receipt) = self.receipt.as_mut() {
            receipt.fail(reason);
        }
        if !self.deposit_failure() {
            if let Some(contribution) = self.command_contribution.take() {
                contribution.complete(ProviderObservationTerminal::Failed);
            }
            self.report(false);
        }
        if let Some(trace) = self.trace.take() {
            trace.fallback(None, "provider_preparation_failed", None);
        }
    }

    pub(super) fn cleanup_failed(&mut self) {
        let reason = if self.cancelled() {
            runmat_accelerate_api::observation::ProviderObservationFailure::Cancelled
        } else {
            runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed
        };
        if let Some(receipt) = self.receipt.as_mut() {
            receipt.fail(reason);
        }
        if !self.deposit_failure() {
            if let Some(contribution) = self.command_contribution.take() {
                contribution.complete(ProviderObservationTerminal::Failed);
            }
            self.report(false);
        }
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

    pub fn finish(mut self, result: Result<&Value, &RuntimeError>) {
        if self.cancelled() {
            if let Some(receipt) = self.receipt.as_mut() {
                receipt.fail(
                    runmat_accelerate_api::observation::ProviderObservationFailure::Cancelled,
                );
            }
            self.feedback = None;
            return;
        }
        if let Some(right) = self.completion.take() {
            let status = if result.is_ok() {
                runmat_runtime::context::preparation::PreparedCallStatus::Succeeded
            } else {
                runmat_runtime::context::preparation::PreparedCallStatus::CompletedFailure
            };
            if status == runmat_runtime::context::preparation::PreparedCallStatus::CompletedFailure
            {
                if let Some(receipt) = self.receipt.as_mut() {
                    receipt.fail(
                        runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
                    );
                }
            }
            let attributable_success = result.as_ref().is_ok_and(|value| {
                if self.provider_selected() {
                    self.receipt.is_some()
                } else {
                    self.host_execution
                        .as_ref()
                        .map_or(self.host_inputs, |proof| proof.is_confirmed())
                        && !runmat_runtime::dispatcher::value_contains_gpu(value)
                }
            });
            if (status
                == runmat_runtime::context::preparation::PreparedCallStatus::CompletedFailure
                || attributable_success)
                && self.feedback.is_some()
            {
                let binding = right.binding();
                right.deposit(Box::new(BackendCompletedCall {
                    binding,
                    feedback: self.feedback.take().expect("checked above"),
                    receipt: if attributable_success && self.provider_selected() {
                        self.receipt.take()
                    } else {
                        None
                    },
                    command_contribution: self.command_contribution.take(),
                    status,
                }));
            }
            // Traces are diagnostic and do not claim a provider completion.
            if let Some(trace) = self.trace.take() {
                match result {
                    Err(_) => trace.fallback(None, "builtin_body_failed", None),
                    Ok(value)
                        if self.provider_selected()
                            && runmat_runtime::dispatcher::value_contains_gpu(value) =>
                    {
                        trace.submitted(PlacementVariant::ProviderOperation, None)
                    }
                    Ok(_) => trace.complete(Some(PlacementVariant::SharedRuntime), None),
                }
            }
            return;
        }
        let Ok(value) = result else {
            // The body completed with an error. Unlike dropping an unpolled or
            // cancelled future, this is an observed operational failure.
            if let Some(receipt) = self.receipt.as_mut() {
                receipt.fail(
                    runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
                );
            }
            if let Some(contribution) = self.command_contribution.take() {
                contribution.complete(ProviderObservationTerminal::Failed);
            }
            self.report(false);
            return;
        };
        let elapsed = duration_ns_saturating(self.started.elapsed()).max(1);
        if self.provider_selected() {
            if let (Some(receipt), Some(feedback)) = (self.receipt.take(), self.feedback.take()) {
                // The receipt independently requires explicit operation-owner
                // route evidence. Resident/host result shape proves nothing.
                crate::placement::publish_provider_feedback(
                    receipt,
                    *feedback,
                    self.command_contribution.take(),
                );
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
            if ticket.runtime.is_cancelled() {
                return;
            }
            ticket.observe(
                duration_ns_saturating(self.started.elapsed()).max(1),
                succeeded,
            );
        }
    }

    fn cancelled(&self) -> bool {
        self.feedback
            .as_ref()
            .is_some_and(|ticket| ticket.runtime.is_cancelled())
    }
}
