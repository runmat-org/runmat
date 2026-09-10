use std::collections::BTreeSet;

use runmat_mir::{MirIndexing, MirOperand, MirOutputTarget, MirPlace, MirStmtKind};
use runmat_native_codegen::NativeInstruction;
use runmat_runtime::native::NativeValueRef;
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::state::HostState;

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
        MirStmtKind::PlaceMutation(mutation) => {
            state.pending_place_mutation = Some(mutation.clone());
            publish_roots(state, instruction, std::slice::from_ref(&mutation.place))?;
            Ok(true)
        }
        MirStmtKind::Assign { place, .. } => {
            let source = input_value(state, instruction, 0)?;
            let mutation = state.pending_place_mutation.take();
            let (delete, allow_init) = if let Some(mutation) = mutation {
                if mutation.place != *place {
                    return Err(NativeExecutorError::Host(
                        "native place-mutation target does not match assignment".into(),
                    ));
                }
                (
                    mutation.kind == runmat_types::PlaceMutationKind::Delete,
                    mutation.creation_policy
                        != runmat_types::AssignmentCreationPolicy::ExistingOnly,
                )
            } else {
                (false, true)
            };
            assign_place(state, place, source, delete, allow_init)?;
            publish_roots(state, instruction, std::slice::from_ref(place))?;
            Ok(true)
        }
        MirStmtKind::MultiAssign { targets, .. } => {
            state.pending_place_mutation = None;
            if instruction.inputs.len() < targets.targets.len() {
                return Err(NativeExecutorError::Host(
                    "native multi-assignment result window is incomplete".into(),
                ));
            }
            for (index, target) in targets.targets.iter().enumerate() {
                let MirOutputTarget::Place(place) = target else {
                    continue;
                };
                let source = input_value(state, instruction, index)?;
                assign_place(state, place, source, false, true)?;
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
            Ok(true)
        }
        MirStmtKind::SequenceAssign { target, .. } => {
            let prepared = state.sequence_assignment_register.take().ok_or_else(|| {
                NativeExecutorError::Host(
                    "native sequence assignment has no produced value sequence".into(),
                )
            })?;
            let values = prepared
                .values
                .into_iter()
                .map(|reference| state.arena.get(reference).cloned())
                .collect::<NativeExecutorResult<Vec<_>>>()?;
            let base = state.arena.get(prepared.destination.base)?.clone();
            let runtime = state.runtime.clone();
            let updated = super::sync::complete(
                &runtime,
                runmat_runtime::object::resolve::store_member_sequence_traced(
                    base,
                    prepared.destination.field,
                    values,
                    Some(&state.function.name),
                ),
                "comma-separated member assignment",
            )?;
            commit_prepared_place_write(state, prepared.destination.writeback, updated)?;
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
    let (writeback, base) = prepare_place_write(state, base_place)?;
    let count = runmat_runtime::object::resolve::member_sequence_cardinality(&base)
        .map_err(NativeExecutorError::from)?;
    let base = state.arena.insert(base);
    Ok((
        PreparedSequenceDestination {
            writeback,
            base,
            field: member,
        },
        count,
    ))
}

fn prepare_place_write(
    state: &mut HostState,
    place: &MirPlace,
) -> NativeExecutorResult<(PreparedPlaceWrite, Value)> {
    let mut segments = Vec::new();
    let root = flatten_place(place, &mut segments)?;
    let root_reference = state.locals.get(root.0).copied().ok_or_else(|| {
        NativeExecutorError::Host("assignment root local is out of bounds".into())
    })?;
    let mut current = state.arena.get(root_reference)?.clone();
    let mut parents = Vec::with_capacity(segments.len());
    for segment in &segments {
        let child = read_segment(state, current.clone(), segment)?;
        parents.push((state.arena.insert(current), segment.clone()));
        current = child;
    }
    Ok((PreparedPlaceWrite { root, parents }, current))
}

fn commit_prepared_place_write(
    state: &mut HostState,
    prepared: PreparedPlaceWrite,
    mut updated: Value,
) -> NativeExecutorResult<()> {
    for (parent, segment) in prepared.parents.into_iter().rev() {
        let parent = state.arena.get(parent)?.clone();
        updated = write_segment(state, parent, &segment, updated, false, true)?;
    }
    let reference = state.arena.insert(updated);
    state.set_local(prepared.root.0, reference)
}

fn assign_place(
    state: &mut HostState,
    place: &MirPlace,
    rhs: Value,
    delete: bool,
    allow_init: bool,
) -> NativeExecutorResult<()> {
    let mut segments = Vec::new();
    let root = flatten_place(place, &mut segments)?;
    if segments.is_empty() {
        let reference = state.arena.insert(rhs);
        return state.set_local(root.0, reference);
    }
    let root_reference = state.locals.get(root.0).copied().ok_or_else(|| {
        NativeExecutorError::Host("assignment root local is out of bounds".into())
    })?;
    let mut current = state.arena.get(root_reference)?.clone();
    let mut parents = Vec::with_capacity(segments.len().saturating_sub(1));
    for segment in &segments[..segments.len() - 1] {
        let child = read_segment(state, current.clone(), segment)?;
        parents.push((current, segment.clone()));
        current = child;
    }
    let mut updated = write_segment(
        state,
        current,
        segments.last().expect("nonempty place segments"),
        rhs,
        delete,
        allow_init,
    )?;
    for (parent, segment) in parents.into_iter().rev() {
        updated = write_segment(state, parent, &segment, updated, false, true)?;
    }
    let reference = state.arena.insert(updated);
    state.set_local(root.0, reference)
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

fn read_segment(
    state: &mut HostState,
    base: Value,
    segment: &PlaceSegment,
) -> NativeExecutorResult<Value> {
    match segment {
        PlaceSegment::Member(member) => super::sync::complete(
            &state.runtime,
            runmat_runtime::object::resolve::load_member_with_context(
                Some(&state.runtime),
                base,
                member.clone(),
                false,
                Some(&state.function.name),
            ),
            "member read",
        ),
        PlaceSegment::DynamicMember(member) => {
            let member = super::operand::materialize_operand(state, member)?;
            let member = String::try_from(&member).map_err(|error| {
                NativeExecutorError::from(runmat_runtime::runtime_error::semantic_error(
                    "DynamicFieldName",
                    error,
                ))
            })?;
            super::sync::complete(
                &state.runtime,
                runmat_runtime::object::resolve::load_member_dynamic_with_context(
                    Some(&state.runtime),
                    base,
                    member,
                    false,
                    Some(&state.function.name),
                ),
                "dynamic member read",
            )
        }
        PlaceSegment::Index(indexing) => {
            let values = super::indexing::read_value(state, base, indexing, 1)?;
            values.into_iter().next().ok_or_else(|| {
                NativeExecutorError::Host("nested indexing did not produce one value".into())
            })
        }
    }
}

fn write_segment(
    state: &mut HostState,
    base: Value,
    segment: &PlaceSegment,
    rhs: Value,
    delete: bool,
    allow_init: bool,
) -> NativeExecutorResult<Value> {
    match segment {
        PlaceSegment::Member(member) => super::sync::complete(
            &state.runtime,
            runmat_runtime::object::resolve::store_member_traced(
                base,
                member.clone(),
                rhs,
                allow_init,
                Some(&state.function.name),
            ),
            "member write",
        ),
        PlaceSegment::DynamicMember(member) => {
            let member = super::operand::materialize_operand(state, member)?;
            let member = String::try_from(&member).map_err(|error| {
                NativeExecutorError::from(runmat_runtime::runtime_error::semantic_error(
                    "DynamicFieldName",
                    error,
                ))
            })?;
            super::sync::complete(
                &state.runtime,
                runmat_runtime::object::resolve::store_member_dynamic_traced(
                    base,
                    member,
                    rhs,
                    allow_init,
                    Some(&state.function.name),
                ),
                "dynamic member write",
            )
        }
        PlaceSegment::Index(indexing) => {
            super::indexing::assign(state, base, indexing, rhs, delete)
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
