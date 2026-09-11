use runmat_mir::{MirPlace, MirSequenceTarget};
use runmat_runtime::sequence::{
    AssignmentStepSpec, PreparedSequenceDestination, SequenceDestinationBuilder,
    SequenceEndpointSpec,
};
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::root_local_value;
use crate::execute::state::HostState;

pub(super) fn prepare_sequence_target(
    state: &mut HostState,
    target: &MirSequenceTarget,
) -> NativeExecutorResult<(runmat_mir::MirLocalId, PreparedSequenceDestination)> {
    let base = target.base();
    let mut path = Vec::new();
    let root = flatten_root(base, &mut path)?;
    let root_value = root_local_value(state, base)?;
    let mut builder = SequenceDestinationBuilder::new(&root_value);
    for place in path {
        let step = assignment_step(state, builder.current(), place)?;
        super::super::super::sync::complete(
            &state.runtime,
            builder.push_step(step, Some(&state.function.name)),
            "native sequence destination preparation",
        )?;
    }
    let endpoint = match target {
        MirSequenceTarget::Member { member, .. } => SequenceEndpointSpec::Member(member.clone()),
        MirSequenceTarget::DynamicMember { member, .. } => {
            let member = super::super::super::operand::materialize_operand(state, member)?;
            SequenceEndpointSpec::Member(runmat_types::MemberName(
                String::try_from(&member).map_err(NativeExecutorError::Host)?,
            ))
        }
        MirSequenceTarget::CellContents { indexing, .. } => SequenceEndpointSpec::CellContents(
            super::super::super::indexing::materialize_index_components(
                state,
                builder.current(),
                indexing,
            )?,
        ),
    };
    let destination = super::super::super::sync::complete(
        &state.runtime,
        builder.finish(endpoint),
        "native sequence destination finalization",
    )?;
    Ok((root, destination))
}

fn assignment_step(
    state: &mut HostState,
    current: &Value,
    place: &MirPlace,
) -> NativeExecutorResult<AssignmentStepSpec> {
    match place {
        MirPlace::Member(_, member) => Ok(AssignmentStepSpec::Member(member.clone())),
        MirPlace::DynamicMember(_, member) => {
            let member = super::super::super::operand::materialize_operand(state, member)?;
            Ok(AssignmentStepSpec::Member(runmat_types::MemberName(
                String::try_from(&member).map_err(NativeExecutorError::Host)?,
            )))
        }
        MirPlace::Index(_, indexing) => {
            let selectors = super::super::super::indexing::materialize_index_components(
                state, current, indexing,
            )?;
            Ok(match indexing.kind {
                runmat_types::IndexKind::Paren => AssignmentStepSpec::Parentheses { selectors },
                runmat_types::IndexKind::Brace => AssignmentStepSpec::Braces(selectors),
            })
        }
        MirPlace::Local(_) | MirPlace::Binding(_) => Err(NativeExecutorError::Host(
            "sequence destination path contains a root as an interior step".into(),
        )),
    }
}

pub(super) fn root_local(place: &MirPlace) -> NativeExecutorResult<runmat_mir::MirLocalId> {
    let mut path = Vec::new();
    flatten_root(place, &mut path)
}

fn flatten_root<'a>(
    place: &'a MirPlace,
    path: &mut Vec<&'a MirPlace>,
) -> NativeExecutorResult<runmat_mir::MirLocalId> {
    match place {
        MirPlace::Local(local) => Ok(*local),
        MirPlace::Binding(binding) => Err(NativeExecutorError::Host(format!(
            "verified Native IR retains legacy MIR binding place {binding:?}"
        ))),
        MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) | MirPlace::Index(base, _) => {
            let root = flatten_root(base, path)?;
            path.push(place);
            Ok(root)
        }
    }
}
