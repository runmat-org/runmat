use std::collections::BTreeMap;

use runmat_mir::{MirOutputTarget, MirOutputTargetList};
use runmat_runtime::native::{NativeSiteRequest, NativeValueRef};
use runmat_runtime::sequence::{DestinationCardinality, DestinationLayout};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::target::PendingSequenceTarget;
use super::{PreparedMultiAssignment, PreparedTarget};
use crate::execute::state::HostState;

pub(in crate::execute) struct PendingMultiPreparation {
    request: NativeSiteRequest,
    targets: MirOutputTargetList,
    next_target: usize,
    prepared: Vec<PreparedTarget>,
    cardinalities: Vec<DestinationCardinality>,
    retained: Vec<NativeValueRef>,
    active: Option<PendingSequenceTarget>,
}

impl PendingMultiPreparation {
    pub(in crate::execute) fn root_references(&self) -> impl Iterator<Item = NativeValueRef> + '_ {
        self.retained.iter().copied().chain(
            self.active
                .iter()
                .flat_map(PendingSequenceTarget::root_references),
        )
    }
}

pub(super) fn prepare(
    state: &mut HostState,
    targets: &MirOutputTargetList,
) -> NativeExecutorResult<PreparedMultiAssignment> {
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("native output preparation has no active rvalue site".into())
    })?;
    let mut pending = match state.pending_multi_preparation.take() {
        Some(pending) if pending.request == request && pending.targets == *targets => pending,
        Some(_) => {
            return Err(NativeExecutorError::Host(
                "pending output preparation does not match its exact resumed site".into(),
            ));
        }
        None => PendingMultiPreparation {
            request,
            targets: targets.clone(),
            next_target: 0,
            prepared: Vec::with_capacity(targets.targets.len()),
            cardinalities: Vec::with_capacity(targets.targets.len()),
            retained: Vec::new(),
            active: None,
        },
    };
    for index in 0..targets.targets.len() {
        let target = &targets.targets[index];
        if index < pending.next_target {
            if matches!(target, MirOutputTarget::Sequence(_)) {
                super::super::continuation::skip_completed_region(state)?;
            }
            continue;
        }
        match target {
            MirOutputTarget::Place(place) => {
                pending.prepared.push(PreparedTarget::Place(place.clone()));
                pending.cardinalities.push(DestinationCardinality::Fixed);
            }
            MirOutputTarget::Discard => {
                pending.prepared.push(PreparedTarget::Discard);
                pending.cardinalities.push(DestinationCardinality::Discard);
            }
            MirOutputTarget::Sequence(target) => {
                let mut active = match pending.active.take() {
                    Some(active) if active.target == *target => active,
                    Some(_) => {
                        return Err(NativeExecutorError::Host(
                            "pending sequence target changed during native preparation".into(),
                        ));
                    }
                    None => PendingSequenceTarget::new(state, target)?,
                };
                match active.advance(state) {
                    Ok(Some((root, destination))) => {
                        destination.visit_retained_values(&mut |value| {
                            pending.retained.push(state.arena.insert(value.clone()));
                        });
                        pending
                            .cardinalities
                            .push(DestinationCardinality::Sequence(destination.cardinality()));
                        pending
                            .prepared
                            .push(PreparedTarget::Sequence { root, destination });
                    }
                    Ok(None) => {
                        return Err(NativeExecutorError::Host(
                            "sequence target preparation ended without a destination".into(),
                        ));
                    }
                    Err(NativeExecutorError::CallSuspended) => {
                        pending.active = Some(active);
                        state.pending_multi_preparation = Some(pending);
                        return Err(NativeExecutorError::CallSuspended);
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        pending.next_target = index + 1;
    }
    Ok(PreparedMultiAssignment {
        targets: pending.prepared,
        layout: DestinationLayout::new(pending.cardinalities)?,
        values: Vec::new(),
        retained: pending.retained,
        next_target: 0,
        roots: BTreeMap::new(),
    })
}
