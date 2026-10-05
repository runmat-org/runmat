use runmat_mir::MirSequenceTarget;
use runmat_runtime::native::{NativeSiteRequest, NativeValueRef};
use runmat_runtime::sequence::PreparedSequenceDestination;
use runmat_value::Value;

use super::super::{HostState, PreparedSequenceAssignment};
use super::{member_call, write_step};
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
                let updated = member_call(
                    state,
                    Box::pin(async move {
                        let value = runtime
                            .scope(runmat_runtime::object::resolve::store_member_sequence_traced_in_context(
                                &runtime,
                                base,
                                field,
                                values,
                                Some(&caller),
                            ))
                            .await?;
                        Ok(vec![value])
                    }),
                )?;
                self.phase = SequencePhase::Writeback(state.arena.insert(updated));
            }
            SequencePhase::Writeback(current) => {
                let Some((parent, segment)) = self.prepared.destination.writeback.parents.last()
                else {
                    return Ok(true);
                };
                let parent = state.arena.get(*parent)?.clone();
                let value = state.arena.get(current)?.clone();
                let updated = write_step(state, parent, segment, value, false, true)?;
                self.prepared.destination.writeback.parents.pop();
                self.phase = SequencePhase::Writeback(state.arena.insert(updated));
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
    let (region, next_step, _) = state.enter_contextual_region();
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
        state.enter_contextual_step(region, step);
        let result = cursor.advance(state);
        let next_step = if result.is_ok() { cursor.step } else { step };
        state.finish_contextual_step(region, step, next_step)?;
        match result {
            Ok(true) => {
                let SequencePhase::Writeback(reference) = cursor.phase else {
                    unreachable!("finished sequence assignment has a writeback result")
                };
                state.finish_contextual_region(region, vec![reference]);
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
    destination: PreparedSequenceDestination,
    root: Value,
    values: Vec<Value>,
) -> NativeExecutorResult<Value> {
    let (region, next_step, _) = state.enter_contextual_region();
    if next_step != 0 {
        return Err(NativeExecutorError::Host(
            "pending prepared sequence target has invalid phase".into(),
        ));
    }
    state.enter_contextual_step(region, 0);
    let runtime = state.runtime.clone();
    let caller = state.function.name.clone();
    let result = member_call(
        state,
        Box::pin(async move {
            let value = runtime
                .scope(destination.assign_in_context(&runtime, root, values, Some(&caller)))
                .await?;
            Ok(vec![value])
        }),
    );
    let next_step = usize::from(result.is_ok());
    state.finish_contextual_step(region, 0, next_step)?;
    if let Ok(value) = &result {
        let reference = state.arena.insert(value.clone());
        state.finish_contextual_region(region, vec![reference]);
    }
    result
}
