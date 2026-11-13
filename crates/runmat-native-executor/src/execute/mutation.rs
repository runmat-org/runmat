use std::collections::BTreeSet;

use runmat_mir::{MirIndexing, MirOperand, MirOutputTarget, MirPlace, MirStmtKind};
use runmat_native_codegen::NativeInstruction;
use runmat_runtime::native::NativeValueRef;
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::state::HostState;

mod continuation;
mod prepared_multi;
pub(super) use continuation::skip_completed_region as skip_prepared_sequence_region;
pub(super) use continuation::PendingAssign;
pub(super) use continuation::PendingPlacePrepare;
pub(super) use continuation::PendingPlainMulti;
pub(super) use continuation::PendingSequenceAssign;
pub(super) use prepared_multi::{has_fixed_end_context, resolve_fixed_end};
pub(super) use prepared_multi::{PendingMultiPreparation, PreparedMultiAssignment};

#[derive(Clone)]
enum PlaceSegment {
    Member(String),
    DynamicMember(MirOperand),
    Index(MirIndexing),
}

pub(super) struct PreparedSequenceDestination {
    writeback: PreparedPlaceWrite,
    base: NativeValueRef,
    field: String,
}

pub(super) struct PreparedSequenceRvalue {
    pub(super) destination: PreparedSequenceDestination,
    pub(super) count: usize,
}

impl PreparedSequenceRvalue {
    pub(super) fn root_references(&self) -> impl Iterator<Item = NativeValueRef> + '_ {
        self.destination
            .writeback
            .parents
            .iter()
            .map(|(value, _)| *value)
            .chain(std::iter::once(self.destination.base))
    }
}

pub(super) struct PreparedSequenceAssignment {
    destination: PreparedSequenceDestination,
    values: Vec<NativeValueRef>,
}

impl PreparedSequenceAssignment {
    pub(super) fn with_values(
        destination: PreparedSequenceDestination,
        values: Vec<NativeValueRef>,
    ) -> NativeExecutorResult<Self> {
        let root_count = destination
            .writeback
            .parents
            .len()
            .checked_add(1)
            .and_then(|count| count.checked_add(values.len()))
            .ok_or_else(|| {
                NativeExecutorError::Host("native sequence root count overflowed".into())
            })?;
        u32::try_from(root_count).map_err(|_| {
            NativeExecutorError::Host(
                "native sequence root count exceeds the supported limit".into(),
            )
        })?;
        Ok(Self {
            destination,
            values,
        })
    }

    pub(super) fn root_references(&self) -> impl Iterator<Item = (u32, NativeValueRef)> + '_ {
        (0_u32..).zip(
            self.destination
                .writeback
                .parents
                .iter()
                .map(|(value, _)| *value)
                .chain(std::iter::once(self.destination.base))
                .chain(self.values.iter().copied()),
        )
    }
}

struct PreparedPlaceWrite {
    root: runmat_mir::MirLocalId,
    parents: Vec<(NativeValueRef, PlaceSegment)>,
}

pub(super) fn execute(
    state: &mut HostState,
    instruction: &NativeInstruction,
    statement: &MirStmtKind,
) -> NativeExecutorResult<bool> {
    match statement {
        MirStmtKind::PrepareFixedAssignment(owner) => {
            owner.validate().map_err(NativeExecutorError::Host)?;
            if state.prepared_multi_assignment.is_some() {
                return Err(NativeExecutorError::Host(
                    "fixed preparation would overwrite a live destination".into(),
                ));
            }
            continuation::validate_creation_policy(
                &owner.assignment.mutation.place,
                owner.assignment.mutation.kind == runmat_types::PlaceMutationKind::Delete,
                owner.assignment.mutation.creation_policy,
            )?;
            let prepared = PreparedMultiAssignment::prepare_fixed(state, owner)?;
            state.prepared_multi_assignment = prepared;
            if let runmat_mir::MirFixedPreparationPhase::EvaluateSelector {
                destination: runmat_mir::MirFixedSelectorDestination::Scalar(local),
                ..
            } = &owner.phase
            {
                let [output] = instruction.outputs.as_slice() else {
                    return Err(NativeExecutorError::Host(
                        "fixed scalar phase has wrong output arity".into(),
                    ));
                };
                let value = state.local_reference(local.0)?;
                state.values.insert(output.value, value);
            } else if !instruction.outputs.is_empty() {
                return Err(NativeExecutorError::Host(
                    "non-scalar fixed phase declares value outputs".into(),
                ));
            }
            Ok(true)
        }
        MirStmtKind::PreparedAssign { preparation, .. } => {
            let mut prepared = state.prepared_multi_assignment.take().ok_or_else(|| {
                NativeExecutorError::Host("fixed assignment has no retained destination".into())
            })?;
            if !prepared.matches_fixed(preparation) {
                return Err(NativeExecutorError::Host(
                    "fixed assignment source occurrence or mutation authority changed".into(),
                ));
            }
            if !prepared.fixed_values_bound() {
                let source = instruction
                    .inputs
                    .first()
                    .and_then(|value| state.values.get(value))
                    .copied()
                    .ok_or_else(|| {
                        NativeExecutorError::Host(
                            "fixed assignment RHS SSA value is unavailable".into(),
                        )
                    })?;
                prepared.bind_fixed_value(source)?;
            }
            if let Err(error) = prepared.commit(state) {
                state.prepared_multi_assignment = Some(prepared);
                return Err(error);
            }
            prepared.release_fixed_original(&mut state.arena)?;
            let root = preparation.mutation.place.root_local().ok_or_else(|| {
                NativeExecutorError::Host("fixed publication has no source root".into())
            })?;
            let [output] = instruction.outputs.as_slice() else {
                return Err(NativeExecutorError::Host(
                    "fixed publication has wrong output arity".into(),
                ));
            };
            let value = state.local_reference(root.0)?;
            state.values.insert(output.value, value);
            Ok(true)
        }
        MirStmtKind::PlaceMutation(mutation) => {
            state.pending_assign = None;
            state.pending_plain_multi = None;
            state.pending_place_mutation = Some(mutation.clone());
            publish_roots(state, instruction, std::slice::from_ref(&mutation.place))?;
            Ok(true)
        }
        MirStmtKind::Assign { place, .. } => {
            let source = input_value(state, instruction, 0)?;
            let mutation = state.pending_place_mutation.as_ref();
            let (delete, creation) = if let Some(mutation) = mutation {
                if mutation.place != *place {
                    return Err(NativeExecutorError::Host(
                        "native place-mutation target does not match assignment".into(),
                    ));
                }
                let delete = mutation.kind == runmat_types::PlaceMutationKind::Delete;
                continuation::validate_creation_policy(place, delete, mutation.creation_policy)?;
                (delete, mutation.creation_policy)
            } else {
                (false, place.assignment_creation_policy(false))
            };
            if place.requires_prepared_assignment() {
                return Err(NativeExecutorError::Host(
                    "expanded assignment requires its canonical prepared producer/consumer".into(),
                ));
            } else {
                continuation::assign(state, place, source, delete, creation)?;
            }
            state.pending_place_mutation = None;
            publish_roots(state, instruction, std::slice::from_ref(place))?;
            Ok(true)
        }
        MirStmtKind::MultiAssign { targets, .. } => {
            state.pending_place_mutation = None;
            if targets
                .targets
                .iter()
                .any(|target| matches!(target, MirOutputTarget::Sequence(_)))
            {
                state.pending_plain_multi = None;
                let mut prepared = state.prepared_multi_assignment.take().ok_or_else(|| {
                    NativeExecutorError::Host(
                        "native sequence-target multi-assignment has no prepared values".into(),
                    )
                })?;
                match prepared.commit(state) {
                    Ok(()) => {}
                    Err(NativeExecutorError::CallSuspended) => {
                        state.prepared_multi_assignment = Some(prepared);
                        return Err(NativeExecutorError::CallSuspended);
                    }
                    Err(error) => return Err(error),
                }
                let places = targets
                    .targets
                    .iter()
                    .filter_map(|target| match target {
                        MirOutputTarget::Place(place) => Some(place),
                        MirOutputTarget::Sequence(target) => Some(target.base()),
                        MirOutputTarget::Discard => None,
                    })
                    .collect::<Vec<_>>();
                publish_roots_from_refs(state, instruction, &places)?;
                return Ok(true);
            }
            if instruction.inputs.len() < targets.targets.len() {
                return Err(NativeExecutorError::Host(
                    "native multi-assignment result window is incomplete".into(),
                ));
            }
            let request = state.current_request.ok_or_else(|| {
                NativeExecutorError::Host(
                    "native multi-assignment has no active statement site".into(),
                )
            })?;
            let next_target = match state.pending_plain_multi.take() {
                Some(pending) if pending.request == request => pending.next_target,
                Some(_) => {
                    return Err(NativeExecutorError::Host(
                        "pending multi-assignment does not match the exact resumed site".into(),
                    ));
                }
                None => 0,
            };
            for (index, target) in targets.targets.iter().enumerate() {
                let MirOutputTarget::Place(place) = target else {
                    continue;
                };
                if index < next_target {
                    continuation::skip_completed(state, place)?;
                    continue;
                }
                let source = input_value(state, instruction, index)?;
                match continuation::assign(
                    state,
                    place,
                    source,
                    false,
                    place.assignment_creation_policy(false),
                ) {
                    Ok(()) => {}
                    Err(NativeExecutorError::CallSuspended) => {
                        state.pending_plain_multi = Some(PendingPlainMulti {
                            request,
                            next_target: index,
                        });
                        return Err(NativeExecutorError::CallSuspended);
                    }
                    Err(error) => return Err(error),
                }
            }
            state.pending_plain_multi = None;
            let places = targets
                .targets
                .iter()
                .filter_map(|target| match target {
                    MirOutputTarget::Place(place) => Some(place),
                    MirOutputTarget::Sequence(target) => Some(target.base()),
                    MirOutputTarget::Discard => None,
                })
                .collect::<Vec<_>>();
            publish_roots_from_refs(state, instruction, &places)?;
            Ok(true)
        }
        MirStmtKind::SequenceAssign { target, .. } => {
            continuation::assign_sequence(state, target)?;
            let base = match target {
                runmat_mir::MirSequenceTarget::Member { base, .. }
                | runmat_mir::MirSequenceTarget::DynamicMember { base, .. }
                | runmat_mir::MirSequenceTarget::CellContents { base, .. } => base,
            };
            publish_roots(state, instruction, std::slice::from_ref(base))?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

pub(super) fn prepare_sequence_assignment(
    state: &mut HostState,
    target: &runmat_mir::MirSequenceTarget,
) -> NativeExecutorResult<(PreparedSequenceDestination, usize)> {
    let (base_place, member) = match target {
        runmat_mir::MirSequenceTarget::Member { base, member } => (base, member.0.clone()),
        runmat_mir::MirSequenceTarget::DynamicMember { base, member } => {
            let member = super::operand::materialize_operand(state, member)?;
            (
                base,
                String::try_from(&member).map_err(NativeExecutorError::Host)?,
            )
        }
        runmat_mir::MirSequenceTarget::CellContents { .. } => {
            return Err(NativeExecutorError::Host(
                "legacy sequence-assignment statements cannot target brace contents".into(),
            ));
        }
    };
    let (writeback, base) = continuation::prepare_place_write(state, base_place)?;
    let count = runmat_runtime::object::resolve::member_sequence_cardinality(&base)
        .map_err(NativeExecutorError::from)?;
    let base = state.arena.insert(base, Some(&state.runtime))?;
    Ok((
        PreparedSequenceDestination {
            writeback,
            base,
            field: member,
        },
        count,
    ))
}

fn flatten_place(
    place: &MirPlace,
    segments: &mut Vec<PlaceSegment>,
) -> NativeExecutorResult<runmat_mir::MirLocalId> {
    match place {
        MirPlace::Local(local) => Ok(*local),
        MirPlace::Binding(binding) => Err(NativeExecutorError::Host(format!(
            "verified Native IR retains legacy MIR binding place {binding:?}"
        ))),
        MirPlace::Member(base, member) => {
            let root = flatten_place(base, segments)?;
            segments.push(PlaceSegment::Member(member.0.clone()));
            Ok(root)
        }
        MirPlace::DynamicMember(base, member) => {
            let root = flatten_place(base, segments)?;
            segments.push(PlaceSegment::DynamicMember(member.clone()));
            Ok(root)
        }
        MirPlace::Index(base, indexing) => {
            let root = flatten_place(base, segments)?;
            segments.push(PlaceSegment::Index(indexing.clone()));
            Ok(root)
        }
    }
}

fn input_value(
    state: &HostState,
    instruction: &NativeInstruction,
    index: usize,
) -> NativeExecutorResult<Value> {
    let value = instruction
        .inputs
        .get(index)
        .and_then(|value| state.values.get(value))
        .copied()
        .ok_or_else(|| NativeExecutorError::Host("statement input value is unavailable".into()))?;
    state.arena.get(value).cloned()
}

fn publish_roots(
    state: &mut HostState,
    instruction: &NativeInstruction,
    places: &[MirPlace],
) -> NativeExecutorResult<()> {
    publish_roots_from_refs(state, instruction, &places.iter().collect::<Vec<_>>())
}

fn publish_roots_from_refs(
    state: &mut HostState,
    instruction: &NativeInstruction,
    places: &[&MirPlace],
) -> NativeExecutorResult<()> {
    let mut seen = BTreeSet::new();
    let roots = places
        .iter()
        .filter_map(|place| root_local(place))
        .filter(|local| seen.insert(*local))
        .collect::<Vec<_>>();
    if roots.len() != instruction.outputs.len() {
        return Err(NativeExecutorError::Host(
            "statement root/output arity does not match Native IR".into(),
        ));
    }
    for (root, output) in roots.iter().zip(&instruction.outputs) {
        let value =
            state.locals.get(root.0).copied().ok_or_else(|| {
                NativeExecutorError::Host("statement root is out of bounds".into())
            })?;
        state.values.insert(output.value, value);
    }
    Ok(())
}

fn root_local(place: &MirPlace) -> Option<runmat_mir::MirLocalId> {
    match place {
        MirPlace::Local(local) => Some(*local),
        MirPlace::Binding(_) => None,
        MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) | MirPlace::Index(base, _) => {
            root_local(base)
        }
    }
}
