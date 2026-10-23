use super::{
    PendingMultiPreparation, PendingSequenceTarget, PreparationTargets, PreparedMultiAssignment,
};
use crate::execute::state::HostState;
use crate::{NativeExecutorError, NativeExecutorResult};
use runmat_mir::{
    MirCallArg, MirFixedPreparation, MirFixedPreparationPhase, MirFixedSelectorDestination,
};

pub(in crate::execute) fn has_fixed_end_context(state: &HostState) -> bool {
    state
        .pending_multi_preparation
        .as_ref()
        .and_then(|pending| pending.active.as_ref())
        .is_some_and(PendingSequenceTarget::has_fixed_end_context)
}

pub(in crate::execute) fn resolve_fixed_end(
    state: &mut HostState,
) -> NativeExecutorResult<runmat_runtime::native::NativeValueRef> {
    let mut pending = state
        .pending_multi_preparation
        .take()
        .ok_or_else(|| NativeExecutorError::Host("fixed End lost pending owner".into()))?;
    let result = pending
        .active
        .as_mut()
        .ok_or_else(|| NativeExecutorError::Host("fixed End lost builder".into()))?
        .resolve_fixed_end(state);
    state.pending_multi_preparation = Some(pending);
    result
}

pub(in crate::execute) fn prepare_fixed(
    state: &mut HostState,
    preparation: &MirFixedPreparation,
) -> NativeExecutorResult<Option<PreparedMultiAssignment>> {
    let owner = &preparation.assignment;
    if matches!(preparation.phase, MirFixedPreparationPhase::CaptureReceiver) {
        if state.pending_multi_preparation.is_some() {
            return Err(NativeExecutorError::Host(
                "fixed receiver would overwrite a pending destination".into(),
            ));
        }
        let request = state.current_request.ok_or_else(|| {
            NativeExecutorError::Host("fixed receiver has no canonical site".into())
        })?;
        let source = super::super::FixedSourceOwner::capture(state, owner)?;
        let original = crate::execute::mutation::continuation::assignment_root_reference_for_place(
            state,
            &owner.mutation.place,
            owner.mutation.creation_policy,
        )?;
        let active = match PendingSequenceTarget::new_fixed(state, source.clone(), original) {
            Ok(active) => active,
            Err(error) => {
                let _ = state.arena.release(original);
                return Err(error);
            }
        };
        state.pending_multi_preparation = Some(PendingMultiPreparation {
            request,
            targets: PreparationTargets::Fixed(source),
            next_target: 0,
            prepared: Vec::new(),
            cardinalities: Vec::new(),
            retained: Vec::new(),
            active: Some(active),
            fixed_original: Some(original),
        });
        return Ok(None);
    }
    let pending = state
        .pending_multi_preparation
        .as_ref()
        .ok_or_else(|| NativeExecutorError::Host("fixed phase has no retained receiver".into()))?;
    if !matches!(&pending.targets, PreparationTargets::Fixed(source) if source.matches(owner)) {
        return Err(NativeExecutorError::Host(
            "fixed phase changed its exact source occurrence".into(),
        ));
    }
    if let MirFixedPreparationPhase::EvaluateSelector {
        path_step,
        component,
        region,
        destination,
    } = &preparation.phase
    {
        state
            .pending_multi_preparation
            .as_mut()
            .and_then(|pending| pending.active.as_mut())
            .ok_or_else(|| {
                NativeExecutorError::Host("fixed selector lost its live builder".into())
            })?
            .begin_fixed_selector(*path_step, *component, region.contains_unscoped_end())?;
        evaluate_selector(state, region, *destination)?;
        return Ok(None);
    }
    let mut pending = state
        .pending_multi_preparation
        .take()
        .expect("validated pending owner");
    let mut active = pending
        .active
        .take()
        .ok_or_else(|| NativeExecutorError::Host("fixed phase lost its builder".into()))?;
    let result = match preparation.phase {
        MirFixedPreparationPhase::BeginIndex { path_step } => {
            active.begin_fixed_index(path_step).map(|()| None)
        }
        MirFixedPreparationPhase::FinishStep { path_step } => {
            active.finish_fixed_step(state, path_step)
        }
        _ => unreachable!("capture and selector phases handled above"),
    };
    match result {
        Ok(Some(prepared)) => {
            let PreparationTargets::Fixed(source) = pending.targets else {
                unreachable!("validated fixed source")
            };
            Ok(Some(PreparedMultiAssignment {
                targets: Vec::new(),
                layout: runmat_runtime::sequence::DestinationLayout::new([])?,
                values: Vec::new(),
                retained: Vec::new(),
                next_target: 0,
                roots: std::collections::BTreeMap::new(),
                fixed: Some(super::super::FixedPreparedAssignment {
                    source,
                    root: prepared.root,
                    original: pending.fixed_original.ok_or_else(|| {
                        NativeExecutorError::Host("fixed phase lost original receiver".into())
                    })?,
                    destination: Some(prepared.destination),
                    rhs: None,
                    updated: None,
                    original_released: false,
                }),
            }))
        }
        result => {
            pending.active = Some(active);
            state.pending_multi_preparation = Some(pending);
            result.map(|_| None)
        }
    }
}

fn evaluate_selector(
    state: &mut HostState,
    region: &runmat_mir::MirCallArgumentRegion,
    destination: MirFixedSelectorDestination,
) -> NativeExecutorResult<()> {
    let (ordinal, next_step, completed) = state.enter_contextual_region()?;
    if completed.is_some() {
        return Ok(());
    }
    if let MirFixedSelectorDestination::Sequence(destination) = destination {
        state.prepare_captured_sequence_slot(destination)?;
    }
    crate::execute::indexing::execute_preparation_steps(state, ordinal, &region.steps, next_step)?;
    let step = region.steps.len();
    state.enter_contextual_step(ordinal, step)?;
    let result = match (destination, &region.argument) {
        (MirFixedSelectorDestination::Scalar(local), MirCallArg::Single(operand)) => {
            let value = crate::execute::operand::evaluate_operand(state, operand)?;
            state.set_local(local.0, value)?;
            Ok(value)
        }
        (MirFixedSelectorDestination::Sequence(to), MirCallArg::CapturedSequence(from)) => {
            state.move_captured_sequence(*from, to)?;
            state
                .pending_multi_preparation
                .as_ref()
                .and_then(|pending| pending.fixed_original)
                .ok_or_else(|| {
                    NativeExecutorError::Host("selector publication lost original receiver".into())
                })
        }
        _ => Err(NativeExecutorError::Host(
            "fixed selector result does not match its typed destination".into(),
        )),
    };
    let finished = state.finish_contextual_step(ordinal, step, step + usize::from(result.is_ok()));
    let value = match (result, finished) {
        (Err(error), _) | (Ok(_), Err(error)) => return Err(error),
        (Ok(value), Ok(())) => value,
    };
    state.finish_contextual_region(ordinal, value);
    Ok(())
}
