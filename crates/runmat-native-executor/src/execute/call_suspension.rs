use std::future::Future;
use std::pin::Pin;

use runmat_runtime::execution::{RootedValue, RootedValueSequence, RootedValues};
use runmat_runtime::native::{NativeSiteRequest, NativeValueRef};
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::state::HostState;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod argument_stage_tests;
mod arguments;
#[cfg(test)]
mod catch_scope_tests;
mod prepared;
mod publication;
mod rooted;
mod semantic_operation;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod roots_tests;
mod sequence;
mod sequence_read;
mod statement_output;
mod short_circuit_truth;
mod subscript;
mod terminator;
pub(super) use arguments::{begin_expanded_call_arguments, take_completed_expanded_call_arguments};
pub(super) use prepared::{begin_prepared_receiver, take_completed_prepared_receiver};
pub(super) use publication::publish;
use rooted::{adopt_rooted_arguments, adopt_rooted_sequence};
pub(super) use rooted::{
    begin_lexical_call, begin_rooted_arguments, begin_rooted_sequence,
    begin_rooted_sequence_with_roots,
};
pub(in crate::execute) use semantic_operation::{
    begin_semantic_operation, begin_semantic_value_operation, settle_completed_semantic_values,
    take_completed_semantic_operation, take_completed_semantic_value_operation,
    SemanticOperationKind,
};
pub(super) use sequence::{begin_sequence_stage, take_completed_sequence_stage, SequenceStage};
pub(super) use sequence_read::{
    begin_sequence_read, take_completed_sequence_read, SequenceReadStage,
};
pub(super) use short_circuit_truth::{
    begin_short_circuit_truth, take_completed_short_circuit_truth, ShortCircuitTruthReady,
    ShortCircuitTruthStage,
};
pub(super) use statement_output::{begin_statement_output, take_completed_statement_output};
pub(super) use subscript::take_completed_subscript_path;
pub(super) use terminator::{
    begin_branch_truth, begin_for_capture, begin_for_next, begin_switch_equality,
    begin_switch_truth, take_completed_terminator, TerminatorCompletion, TerminatorReady,
    TerminatorReadyStage,
};

pub(super) type CompletionFuture =
    Pin<Box<dyn Future<Output = NativeExecutorResult<CallCompletion>> + 'static>>;

pub(super) fn scope_completion(
    state: &HostState,
    future: impl Future<Output = NativeExecutorResult<CallCompletion>> + 'static,
) -> CompletionFuture {
    Box::pin(
        state
            .runtime
            .scope_active_catch_shared(state.active_catch_exception(), future),
    )
}

pub(super) enum CallCompletion {
    RootedSequence {
        sequence: RootedValueSequence,
        requested_outputs: usize,
    },
    RootedArguments(RootedValues),
    RootedExpandedCallArguments(RootedValues),
    Lexical {
        result: runmat_runtime::call::lexical::RootedLexicalCallResult,
        requested_outputs: usize,
    },
    SemanticOperation {
        kind: SemanticOperationKind,
        output: SemanticOperationOutput,
    },
    SequenceRead {
        stage: SequenceReadStage,
        sequence: RootedValueSequence,
    },
    PreparedReceiver(Box<runmat_runtime::object::protocol::PreparedSubscriptReceiver>),
    SequenceStage(Box<SequenceStage>),
    ShortCircuitTruth(ShortCircuitTruthReady),
    StatementOutput,
    Terminator(Box<TerminatorCompletion>),
}

pub(super) struct PendingCall {
    pub(super) continuation: u64,
    pub(super) generation: u64,
    pub(super) request: NativeSiteRequest,
    pub(super) embedded: Option<super::state::EmbeddedOperationIdentity>,
    pub(super) retained_roots: Vec<NativeValueRef>,
    pub(super) future: CompletionFuture,
}

pub(super) struct CompletedCall {
    request: NativeSiteRequest,
    embedded: Option<super::state::EmbeddedOperationIdentity>,
    payload: CompletedPayload,
    // This root-bearing value is kept outside the tagged payload so ordinary
    // completions do not inherit its large inline enum representation.
    sequence_read: Option<RootedValueSequence>,
    // Semantic operations return their exactly rooted owned values directly to
    // the resumed consumer. Keeping them out of the arena avoids a payload
    // clone merely to cross the native suspension boundary.
    semantic_output: Option<SemanticOperationOutput>,
}

pub(super) enum SemanticOperationOutput {
    Values(RootedValues),
    Value(RootedValue),
}

pub(super) enum CompletedPayload {
    Values(Vec<NativeValueRef>),
    ExpandedCallArguments(Vec<NativeValueRef>),
    SequenceRead(SequenceReadStage),
    PreparedReceiver {
        prepared: Box<runmat_runtime::object::protocol::PreparedSubscriptReceiver>,
        receiver: NativeValueRef,
    },
    SequenceStage {
        stage: Box<SequenceStage>,
        roots: Vec<NativeValueRef>,
    },
    ShortCircuitTruth(ShortCircuitTruthReady),
    StatementOutput,
    SemanticOperation(SemanticOperationKind),
    Terminator(Box<TerminatorReady>),
}

pub(super) fn own_output_storage(
    values: runmat_value::OutputListStorage,
) -> NativeExecutorResult<runmat_value::OwnedOutputListValues> {
    values.try_into_owned().map_err(|_| {
        NativeExecutorError::Host("native output adoption requires unique canonical storage".into())
    })
}

impl CompletedCall {
    fn new(
        request: NativeSiteRequest,
        embedded: Option<super::state::EmbeddedOperationIdentity>,
        payload: CompletedPayload,
        sequence_read: Option<RootedValueSequence>,
        semantic_output: Option<SemanticOperationOutput>,
    ) -> NativeExecutorResult<Self> {
        if matches!(&payload, CompletedPayload::SequenceRead(_)) != sequence_read.is_some() {
            return Err(NativeExecutorError::Host(
                "completed sequence read has no exact rooted value".into(),
            ));
        }
        if matches!(&payload, CompletedPayload::SemanticOperation(_)) != semantic_output.is_some() {
            return Err(NativeExecutorError::Host(
                "completed semantic operation has no exact rooted value owner".into(),
            ));
        }
        Ok(Self {
            request,
            embedded,
            payload,
            sequence_read,
            semantic_output,
        })
    }

    fn into_sequence_read(self) -> NativeExecutorResult<RootedValueSequence> {
        match (self.payload, self.sequence_read) {
            (CompletedPayload::SequenceRead(_), Some(sequence)) => Ok(sequence),
            _ => Err(NativeExecutorError::Host(
                "completed sequence read lost its rooted value".into(),
            )),
        }
    }
}

pub(super) fn take_completed(
    state: &mut HostState,
    embedded: Option<&super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<Option<Vec<Value>>> {
    if let Some(embedded) = embedded {
        if let Some(outputs) = state.cached_embedded_call(embedded)? {
            return Ok(Some(outputs));
        }
    }
    let Some(completed) = state.completed_call.take() else {
        return Ok(None);
    };
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("completed native call has no active site".into())
    })?;
    if completed.request != request || completed.embedded.as_ref() != embedded {
        let descendant = completed.request == request
            && completed
                .embedded
                .as_ref()
                .is_some_and(|identity| identity.is_descendant_of(embedded));
        state.completed_call = Some(completed);
        if descendant {
            // The enclosing call is re-entering its argument evaluation. Only
            // the exact child operation may consume this completed payload.
            return Ok(None);
        }
        return Err(NativeExecutorError::Host(
            "completed native call does not match the exact resumed operation".into(),
        ));
    }
    if matches!(
        &completed.payload,
        CompletedPayload::ExpandedCallArguments(_)
    ) {
        // The enclosing call and its argument expansion may both have no
        // contextual-region identity. The typed payload, not an optional
        // embedded identity, prevents arguments from becoming call outputs.
        state.completed_call = Some(completed);
        return Ok(None);
    }
    if matches!(&completed.payload, CompletedPayload::SemanticOperation(_)) {
        // An indexing site may resume either its built-in semantic operation
        // or an object protocol call. Leave the typed semantic completion for
        // its exact consumer rather than treating it as call output.
        state.completed_call = Some(completed);
        return Ok(None);
    }
    let CompletedPayload::Values(outputs) = &completed.payload else {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed native call payload does not match the exact resumed operation".into(),
        ));
    };
    if let Some(embedded) = embedded {
        state.restore_embedded_call_regions(embedded);
    }
    outputs
        .iter()
        .map(|reference| state.arena.get(*reference).cloned())
        .collect::<NativeExecutorResult<Vec<_>>>()
        .map(Some)
}

pub(super) fn completed_roots(state: &HostState) -> impl Iterator<Item = NativeValueRef> + '_ {
    state
        .completed_call
        .iter()
        .flat_map(|completed| match &completed.payload {
            CompletedPayload::Values(outputs) => outputs.clone(),
            CompletedPayload::ExpandedCallArguments(arguments) => arguments.clone(),
            CompletedPayload::SequenceRead(_) => Vec::new(),
            CompletedPayload::PreparedReceiver { receiver, .. } => vec![*receiver],
            CompletedPayload::SequenceStage { roots, .. } => roots.clone(),
            CompletedPayload::ShortCircuitTruth(_) => Vec::new(),
            CompletedPayload::StatementOutput => Vec::new(),
            CompletedPayload::SemanticOperation(_) => Vec::new(),
            CompletedPayload::Terminator(stage) => stage.arena_roots(),
        })
}

pub(super) async fn complete(
    state: &mut HostState,
    continuation: u64,
    generation: u64,
) -> NativeExecutorResult<NativeSiteRequest> {
    validate_pending(state, continuation, generation)?;
    let retained = state
        .pending_call
        .as_ref()
        .expect("validated pending call")
        .retained_roots
        .iter()
        .map(|reference| state.arena.get(*reference))
        .collect::<NativeExecutorResult<Vec<_>>>()?;
    let retained_lease =
        runmat_runtime::execution::RootedValueLease::for_values(retained.into_iter())
            .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
    let pending = state
        .pending_call
        .take()
        .ok_or_else(|| NativeExecutorError::Host("native invocation has no pending call".into()))?;
    let PendingCall {
        request,
        embedded,
        future,
        ..
    } = pending;
    let runtime = state.runtime.clone();
    let completion =
        match runmat_runtime::context::run_selected_callback_continuation(&runtime, async move {
            let _retained_lease = retained_lease;
            Ok::<_, runmat_runtime::RuntimeError>(future.await)
        })
        .await
        {
            Ok(completion) => completion,
            Err(error) => Err(NativeExecutorError::from(error)),
        };
    // A For iterator is moved into the owned future to avoid borrowing
    // HostState across await. Restore that exact record even if workspace
    // reconciliation or column materialization fails.
    let mut sequence_read = None;
    let mut semantic_output = None;
    let payload = match completion {
        Ok(CallCompletion::Terminator(stage)) if stage.is_for_next() => {
            let reconciliation = state.reconcile_external_workspace();
            let restored = stage.restore_for_next(state);
            reconciliation?;
            let next = restored?.map_err(NativeExecutorError::from)?;
            let next = next.map(|value| value.with_owned(|value| state.arena.insert(value)));
            CompletedPayload::Terminator(Box::new(TerminatorReady::for_next(next)))
        }
        completion => {
            let completion =
                super::operation_completion::CompletedOperation::new(completion).finish(state)?;
            match completion {
                CallCompletion::RootedSequence {
                    sequence,
                    requested_outputs,
                } => CompletedPayload::Values(adopt_rooted_sequence(
                    state,
                    sequence,
                    requested_outputs,
                )?),
                CallCompletion::RootedArguments(arguments) => {
                    CompletedPayload::Values(adopt_rooted_arguments(state, arguments)?)
                }
                CallCompletion::RootedExpandedCallArguments(arguments) => {
                    CompletedPayload::ExpandedCallArguments(adopt_rooted_arguments(
                        state, arguments,
                    )?)
                }
                CallCompletion::Lexical {
                    result,
                    requested_outputs,
                } => {
                    let outputs = result.with_owned(|result| {
                        state.apply_lexical_captures(result.captures)?;
                        super::call::normalize_sequence_outputs(result.outputs, requested_outputs)
                            .and_then(|values| {
                                Ok(own_output_storage(values)?
                                    .into_iter()
                                    .map(|value| state.arena.insert(value))
                                    .collect())
                            })
                    })?;
                    CompletedPayload::Values(outputs)
                }
                CallCompletion::SemanticOperation { kind, output } => {
                    semantic_output = Some(output);
                    CompletedPayload::SemanticOperation(kind)
                }
                CallCompletion::SequenceRead { stage, sequence } => {
                    sequence_read = Some(sequence);
                    CompletedPayload::SequenceRead(stage)
                }
                CallCompletion::PreparedReceiver(prepared) => {
                    let receiver = state.arena.insert(prepared.receiver().clone());
                    CompletedPayload::PreparedReceiver { prepared, receiver }
                }
                CallCompletion::SequenceStage(stage) => {
                    let mut roots = Vec::new();
                    stage.visit_retained_values(&mut |value| {
                        roots.push(state.arena.insert(value.clone()));
                    });
                    CompletedPayload::SequenceStage { stage, roots }
                }
                CallCompletion::ShortCircuitTruth(truth) => {
                    CompletedPayload::ShortCircuitTruth(truth)
                }
                CallCompletion::StatementOutput => CompletedPayload::StatementOutput,
                CallCompletion::Terminator(stage) => {
                    CompletedPayload::Terminator(Box::new(stage.finish()?))
                }
            }
        }
    };
    state.completed_call = Some(CompletedCall::new(
        request,
        embedded,
        payload,
        sequence_read,
        semantic_output,
    )?);
    Ok(request)
}

pub(super) fn validate_pending(
    state: &HostState,
    continuation: u64,
    generation: u64,
) -> NativeExecutorResult<()> {
    let pending = state
        .pending_call
        .as_ref()
        .ok_or_else(|| NativeExecutorError::Host("native invocation has no pending call".into()))?;
    if pending.continuation != continuation || pending.generation != generation {
        return Err(NativeExecutorError::Host(
            "native call continuation identity is stale or mismatched".into(),
        ));
    }
    Ok(())
}
