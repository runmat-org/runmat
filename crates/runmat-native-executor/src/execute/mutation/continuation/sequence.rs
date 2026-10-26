use runmat_mir::MirSequenceTarget;
use runmat_runtime::native::{NativeSiteRequest, NativeValueRef};
use runmat_runtime::sequence::PreparedSequenceDestination;
use runmat_value::Value;

use super::super::{HostState, PreparedSequenceAssignment};
use super::{member_call, single_member_sequence, write_step};
use crate::{NativeExecutorError, NativeExecutorResult};

#[derive(Clone, Copy)]
enum SequencePhase {
    Member,
    Writeback(NativeValueRef),
}

pub(in crate::execute) struct PendingSequenceAssign {
    request: NativeSiteRequest,
    target: MirSequenceTarget,
    region: usize,
    step: usize,
    prepared: PreparedSequenceAssignment,
    phase: SequencePhase,
}

impl PendingSequenceAssign {
    pub(in crate::execute) fn root_references(&self) -> impl Iterator<Item = NativeValueRef> + '_ {
        self.prepared
            .root_references()
            .map(|(_, reference)| reference)
            .chain(match self.phase {
                SequencePhase::Member => None,
                SequencePhase::Writeback(value) => Some(value),
            })
    }

    fn advance(&mut self, state: &mut HostState) -> NativeExecutorResult<bool> {
        match self.phase {
            SequencePhase::Member => {
                let values = self
                    .prepared
                    .values
                    .iter()
                    .map(|reference| state.arena.get(*reference).cloned())
                    .collect::<NativeExecutorResult<Vec<_>>>()?;
                let base = state.arena.get(self.prepared.destination.base)?.clone();
                let field = self.prepared.destination.field.clone();
                let runtime = state.runtime.clone();
                let caller = state.function.name.clone();
                let access = super::super::super::subscript_path::frame_access(state)?;
                let updated = member_call(state, move || async move {
                    let value = runtime
                            .scope(runmat_runtime::object::resolve::store_member_sequence_traced_with_access(
                                &runtime,
                                base,
                                field,
                                values,
                                Some(&caller),
                                &access,
                            ))
                            .await?;
                    single_member_sequence(value).map_err(|error| *error)
                })?;
                self.phase =
                    SequencePhase::Writeback(state.arena.insert(updated, Some(&state.runtime))?);
            }
            SequencePhase::Writeback(current) => {
                let Some((parent, segment)) = self.prepared.destination.writeback.parents.last()
                else {
                    return Ok(true);
                };
                let parent = state.arena.get(*parent)?.clone();
                let value = state.arena.get(current)?.clone();
                let updated = write_step(state, parent, segment, value, false)?;
                self.prepared.destination.writeback.parents.pop();
                self.phase =
                    SequencePhase::Writeback(state.arena.insert(updated, Some(&state.runtime))?);
            }
        }
        self.step = self.step.checked_add(1).ok_or_else(|| {
            NativeExecutorError::Host("native sequence mutation step ordinal overflowed".into())
        })?;
        Ok(matches!(self.phase, SequencePhase::Writeback(_))
            && self.prepared.destination.writeback.parents.is_empty())
    }
}

pub(in crate::execute) fn assign_sequence(
    state: &mut HostState,
    target: &MirSequenceTarget,
) -> NativeExecutorResult<()> {
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("native sequence mutation has no active statement site".into())
    })?;
    let (region, next_step, _) = state.enter_contextual_region()?;
    let mut cursor = match state.pending_sequence_assign.take() {
        Some(cursor)
            if cursor.request == request
                && cursor.region == region
                && cursor.step == next_step
                && cursor.target == *target =>
        {
            cursor
        }
        Some(_) => {
            return Err(NativeExecutorError::Host(
                "pending sequence assignment does not match the exact resumed site".into(),
            ));
        }
        None => PendingSequenceAssign {
            request,
            target: target.clone(),
            region,
            step: 0,
            prepared: state.sequence_assignment_register.take().ok_or_else(|| {
                NativeExecutorError::Host(
                    "native sequence assignment has no produced value sequence".into(),
                )
            })?,
            phase: SequencePhase::Member,
        },
    };
    loop {
        let step = cursor.step;
        state.enter_contextual_step(region, step)?;
        let result = cursor.advance(state);
        let next_step = if result.is_ok() { cursor.step } else { step };
        state.finish_contextual_step(region, step, next_step)?;
        match result {
            Ok(true) => {
                let SequencePhase::Writeback(reference) = cursor.phase else {
                    unreachable!("finished sequence assignment has a writeback result")
                };
                state.finish_contextual_region(region, reference);
                return state.set_local(cursor.prepared.destination.writeback.root.0, reference);
            }
            Ok(false) => {}
            Err(NativeExecutorError::CallSuspended) => {
                state.pending_sequence_assign = Some(cursor);
                return Err(NativeExecutorError::CallSuspended);
            }
            Err(error) => return Err(error),
        }
    }
}

pub(in crate::execute) fn assign_prepared_sequence(
    state: &mut HostState,
    destination: &mut Option<PreparedSequenceDestination>,
    root: Value,
    values: Vec<Value>,
) -> NativeExecutorResult<Value> {
    let (region, next_step, _) = state.enter_contextual_region()?;
    if next_step != 0 {
        return Err(NativeExecutorError::Host(
            "pending prepared sequence target has invalid phase".into(),
        ));
    }
    state.enter_contextual_step(region, 0)?;
    let embedded = state.enter_embedded_call()?;
    let result = (|| {
        use super::super::super::call_suspension as suspension;
        let kind = suspension::SemanticOperationKind::IndexedAssignment;
        if let Some(value) =
            suspension::take_completed_semantic_value_operation(state, embedded.as_ref(), kind)?
        {
            return Ok(value.with_owned(|value| value));
        }
        if destination.is_none() {
            return Err(NativeExecutorError::Host(
                "prepared sequence callback lost its move-only destination".into(),
            ));
        }
        let request = state.current_request.ok_or_else(|| {
            NativeExecutorError::Host("prepared sequence callback has no source site".into())
        })?;
        let runtime = state.runtime.clone();
        let admission = runtime.clone();
        let catch = state.active_catch_exception();
        let functions = std::sync::Arc::clone(&state.functions);
        let future = runmat_runtime::execution::admit_native_hostcall_with(
            &admission,
            fixed_allocation_cause,
            || {
                let destination = destination
                    .take()
                    .expect("validated prepared sequence destination");
                runtime
                    .clone()
                    .scope_active_catch_unboxed_shared(catch, async move {
                        let caller = functions
                            .iter()
                            .find(|function| function.portable_id.0 == request.function)
                            .ok_or_else(|| {
                                NativeExecutorError::Host(
                                    "prepared sequence callback lost caller metadata".into(),
                                )
                            })?;
                        let value = destination
                            .assign_in_context(&runtime, root, values, Some(&caller.name))
                            .await?;
                        Ok(suspension::CallCompletion::SemanticOperation {
                            kind,
                            output: suspension::SemanticOperationOutput::Value(value),
                        })
                    })
            },
        )?;
        let (continuation, generation) = state.next_suspension_identity()?;
        state.pending_call = Some(suspension::PendingCall {
            continuation,
            generation,
            request,
            embedded: embedded.clone(),
            retained_roots: Vec::new(),
            future,
        });
        Err(NativeExecutorError::CallSuspended)
    })();
    let finished = state.finish_embedded_call(embedded.as_ref());
    let result = match (result, finished) {
        (Err(error), _) | (Ok(_), Err(error)) => Err(error),
        (Ok(value), Ok(())) => Ok(value),
    };
    let next_step = usize::from(result.is_ok());
    state.finish_contextual_step(region, 0, next_step)?;
    if let Ok(value) = &result {
        let reference = state.arena.insert(value.clone(), Some(&state.runtime))?;
        state.finish_contextual_region(region, reference);
    }
    result
}

/// A fixed destination and its RHS are move-only owners. A suspended callback
/// retains the actual destination; resumption consumes its completion before
/// attempting to construct any replacement future or arguments.
pub(in crate::execute) fn assign_prepared_fixed(
    state: &mut HostState,
    destination: &mut Option<PreparedSequenceDestination>,
    original: NativeValueRef,
    rhs: NativeValueRef,
) -> NativeExecutorResult<NativeValueRef> {
    let (region, next_step, _) = state.enter_contextual_region()?;
    if next_step != 0 {
        return Err(NativeExecutorError::Host(
            "fixed assignment callback has an invalid phase".into(),
        ));
    }
    state.enter_contextual_step(region, 0)?;
    let embedded = state.enter_embedded_call()?;
    let result = (|| {
        use super::super::super::call_suspension as suspension;
        let kind = suspension::SemanticOperationKind::IndexedAssignment;
        match suspension::take_completed_semantic_value_operation(state, embedded.as_ref(), kind)? {
            Some(value) => {
                value.with_owned(|value| state.arena.insert(value, Some(&state.runtime)))
            }
            None => {
                if destination.is_none() {
                    return Err(NativeExecutorError::Host(
                        "pending fixed assignment lost its move-only destination".into(),
                    ));
                }
                state.arena.get(original)?;
                state.arena.get(rhs)?;
                let request = state.current_request.ok_or_else(|| {
                    NativeExecutorError::Host("fixed callback has no source site".into())
                })?;
                let runtime = state.runtime.clone();
                let admission = runtime.clone();
                let catch = state.active_catch_exception();
                let functions = std::sync::Arc::clone(&state.functions);
                let future = runmat_runtime::execution::admit_native_hostcall_with(
                    &admission,
                    fixed_allocation_cause,
                    || {
                        // Frame-local references were validated above, and
                        // admission cannot mutate this exclusively borrowed
                        // frame. Materialize owned inputs only after admission.
                        let destination = destination.take().expect("validated fixed destination");
                        let root = state
                            .arena
                            .get(original)
                            .expect("validated original generation")
                            .clone();
                        let rhs = state
                            .arena
                            .get(rhs)
                            .expect("validated RHS generation")
                            .clone();
                        runtime
                            .clone()
                            .scope_active_catch_unboxed_shared(catch, async move {
                                let caller = functions
                                    .iter()
                                    .find(|function| function.portable_id.0 == request.function)
                                    .ok_or_else(|| {
                                        NativeExecutorError::Host(
                                            "fixed callback lost canonical caller metadata".into(),
                                        )
                                    })?;
                                let value = destination
                                    .assign_fixed_in_context(
                                        &runtime,
                                        root,
                                        rhs,
                                        Some(&caller.name),
                                    )
                                    .await?;
                                Ok(suspension::CallCompletion::SemanticOperation {
                                    kind,
                                    output: suspension::SemanticOperationOutput::Value(value),
                                })
                            })
                    },
                )?;
                let (continuation, generation) = state.next_suspension_identity()?;
                state.pending_call = Some(suspension::PendingCall {
                    continuation,
                    generation,
                    request,
                    embedded: embedded.clone(),
                    retained_roots: Vec::new(),
                    future,
                });
                Err(NativeExecutorError::CallSuspended)
            }
        }
    })();
    let finished = state.finish_embedded_call(embedded.as_ref());
    let result = match (result, finished) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(value), Ok(())) => Ok(value),
    };
    state.finish_contextual_step(region, 0, usize::from(result.is_ok()))?;
    if let Ok(reference) = &result {
        state.finish_contextual_region(region, *reference);
    }
    result
}

fn fixed_allocation_cause(
    source: runmat_runtime::context::host_allocation::HostAllocationError,
) -> NativeExecutorError {
    runmat_runtime::context::host_allocation::host_allocation_runtime_error(
        None,
        "RunMat:Native:FixedAssignment",
        "fixed assignment continuation admission failed",
        source,
    )
    .into()
}
