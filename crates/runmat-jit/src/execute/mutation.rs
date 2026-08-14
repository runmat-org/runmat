use std::collections::BTreeSet;

use futures::executor::block_on;
use runmat_mir::{MirIndexing, MirOperand, MirOutputTarget, MirPlace, MirStmtKind};
use runmat_native_codegen::NativeInstruction;
use runmat_value::Value;

use crate::{JitError, JitResult};

use super::state::HostState;

#[derive(Clone)]
enum PlaceSegment {
    Member(String),
    DynamicMember(MirOperand),
    Index(MirIndexing),
}

pub(super) fn execute(
    state: &mut HostState,
    instruction: &NativeInstruction,
    statement: &MirStmtKind,
) -> JitResult<bool> {
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
                    return Err(JitError::Host(
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
                return Err(JitError::Host(
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
                    MirOutputTarget::Discard => None,
                })
                .collect::<Vec<_>>();
            publish_roots_from_refs(state, instruction, &places)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn assign_place(
    state: &mut HostState,
    place: &MirPlace,
    rhs: Value,
    delete: bool,
    allow_init: bool,
) -> JitResult<()> {
    let mut segments = Vec::new();
    let root = flatten_place(place, &mut segments)?;
    if segments.is_empty() {
        let reference = state.arena.insert(rhs);
        return state.set_local(root.0, reference);
    }
    let root_reference = state
        .locals
        .get(root.0)
        .copied()
        .ok_or_else(|| JitError::Host("assignment root local is out of bounds".into()))?;
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
) -> JitResult<runmat_mir::MirLocalId> {
    match place {
        MirPlace::Local(local) => Ok(*local),
        MirPlace::Binding(binding) => Err(JitError::UnsupportedSite(format!(
            "legacy MIR binding place {binding:?} has no native local"
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

fn read_segment(state: &mut HostState, base: Value, segment: &PlaceSegment) -> JitResult<Value> {
    match segment {
        PlaceSegment::Member(member) => block_on(state.runtime.scope(
            runmat_runtime::object::resolve::load_member(
                base,
                member.clone(),
                false,
                Some(&state.function.name),
            ),
        ))
        .map_err(JitError::from),
        PlaceSegment::DynamicMember(member) => {
            let member = super::operand::materialize_operand(state, member)?;
            let member = String::try_from(&member).map_err(|error| {
                JitError::from(runmat_runtime::runtime_error::semantic_error(
                    "DynamicFieldName",
                    error,
                ))
            })?;
            block_on(
                state
                    .runtime
                    .scope(runmat_runtime::object::resolve::load_member_dynamic(
                        base,
                        member,
                        false,
                        Some(&state.function.name),
                    )),
            )
            .map_err(JitError::from)
        }
        PlaceSegment::Index(indexing) => {
            let values = super::indexing::read_value(state, base, indexing, 1)?;
            values
                .into_iter()
                .next()
                .ok_or_else(|| JitError::Host("nested indexing did not produce one value".into()))
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
) -> JitResult<Value> {
    match segment {
        PlaceSegment::Member(member) => block_on(state.runtime.scope(
            runmat_runtime::object::resolve::store_member_traced(
                base,
                member.clone(),
                rhs,
                allow_init,
                Some(&state.function.name),
            ),
        ))
        .map_err(JitError::from),
        PlaceSegment::DynamicMember(member) => {
            let member = super::operand::materialize_operand(state, member)?;
            let member = String::try_from(&member).map_err(|error| {
                JitError::from(runmat_runtime::runtime_error::semantic_error(
                    "DynamicFieldName",
                    error,
                ))
            })?;
            block_on(state.runtime.scope(
                runmat_runtime::object::resolve::store_member_dynamic_traced(
                    base,
                    member,
                    rhs,
                    allow_init,
                    Some(&state.function.name),
                ),
            ))
            .map_err(JitError::from)
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
) -> JitResult<Value> {
    let value = instruction
        .inputs
        .get(index)
        .and_then(|value| state.values.get(value))
        .copied()
        .ok_or_else(|| JitError::Host("statement input value is unavailable".into()))?;
    state.arena.get(value).cloned()
}

fn publish_roots(
    state: &mut HostState,
    instruction: &NativeInstruction,
    places: &[MirPlace],
) -> JitResult<()> {
    publish_roots_from_refs(state, instruction, &places.iter().collect::<Vec<_>>())
}

fn publish_roots_from_refs(
    state: &mut HostState,
    instruction: &NativeInstruction,
    places: &[&MirPlace],
) -> JitResult<()> {
    let mut seen = BTreeSet::new();
    let roots = places
        .iter()
        .filter_map(|place| root_local(place))
        .filter(|local| seen.insert(*local))
        .collect::<Vec<_>>();
    if roots.len() != instruction.outputs.len() {
        return Err(JitError::Host(
            "statement root/output arity does not match Native IR".into(),
        ));
    }
    for (root, output) in roots.iter().zip(&instruction.outputs) {
        let value = state
            .locals
            .get(root.0)
            .copied()
            .ok_or_else(|| JitError::Host("statement root is out of bounds".into()))?;
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
