use std::collections::HashMap;

use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::execution::{RootedValue, RootedValueLease, RootedValueSequence};
use runmat_runtime::sequence::ResolveValueSequence;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::super::{PreparedOutputTarget, SequenceState};
use super::support::{empty_cell_root, root};

pub(super) async fn commit(
    stack: &mut Vec<Value>,
    vars: &mut InterpreterVars,
    state: &mut SequenceState,
    retained_outputs: usize,
    authority: &super::super::slot_authority::AssignmentSlotAuthority<'_>,
    before_overwrite: &mut dyn FnMut(usize),
    after_store: &mut dyn FnMut(usize),
) -> Result<(), RuntimeError> {
    let layout = state.output_layout()?;
    let source_roots = state
        .assignment
        .and_then(|window| {
            window
                .start
                .checked_add(window.len)
                .and_then(|end| stack.get(window.start..end))
        })
        .map(|values| RootedValueLease::for_values(values.iter()))
        .transpose()
        .map_err(|error| error.into_runtime_error())?;
    let sequence =
        RootedValueSequence::new(super::super::object::take_sequence_register(stack, state)?)
            .map_err(|error| error.into_runtime_error())?;
    drop(source_roots);
    let (partitions, _partition_roots) = sequence.with_owned_sequence(|sequence| {
        let values = sequence.resolve(
            runmat_types::SequenceUse::SelectDestinationCardinality,
            runmat_runtime::sequence::SequenceResolutionContext::destination_cardinality(
                layout.total(),
            ),
        )?;
        let partitions = layout.distribute(values)?;
        let roots = RootedValueLease::for_values(partitions.iter().flatten())
            .map_err(|error| error.into_runtime_error())?;
        Ok::<_, RuntimeError>((partitions, roots))
    })?;
    let targets = state.take_output_targets();
    let mut retained = Vec::with_capacity(retained_outputs);
    let mut roots = Vec::<(usize, Option<RootedValue>)>::new();
    let mut root_positions = HashMap::<usize, usize>::new();
    for (target, values) in targets.into_iter().zip(partitions) {
        match target {
            PreparedOutputTarget::Fixed | PreparedOutputTarget::Discard => retained.extend(values),
            PreparedOutputTarget::Sequence {
                root_slot,
                endpoint,
                absent_root,
                root_creation,
            } => {
                let position = match root_positions.get(&root_slot).copied() {
                    Some(position) => position,
                    None => {
                        // Selectors and the RHS may have called code that changed
                        // a global, persistent, or shared lexical root. Resolve
                        // that binding at commit, not from Begin's old slot copy.
                        let absent_now = authority.refresh(vars, root_slot)?.is_absent();
                        let position = roots.len();
                        roots.push((
                            root_slot,
                            Some(
                                RootedValue::new(
                                    if absent_now
                                        && root_creation
                                            == runmat_mir::SequenceRootCreation::EmptyCell
                                    {
                                        absent_root
                                            .map(|root| *root)
                                            .unwrap_or_else(empty_cell_root)
                                    } else {
                                        root(vars, root_slot)?.clone()
                                    },
                                )
                                .map_err(|error| error.into_runtime_error())?,
                            ),
                        ));
                        root_positions.insert(root_slot, position);
                        position
                    }
                };
                let root_value = roots[position].1.take().ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "RunMat:CommaSeparatedListState",
                        "prepared output root is unavailable during assignment",
                    )
                })?;
                let updated = endpoint
                    .assign(
                        root_value.value().clone(),
                        values,
                        Some(authority.current_function_name),
                    )
                    .await?;
                roots[position].1 = Some(updated);
            }
        }
    }
    if retained.len() != retained_outputs {
        return Err(crate::interpreter::errors::mex(
            "RunMat:CommaSeparatedListState",
            "prepared output assignment retained an unexpected value count",
        ));
    }
    for (slot, value) in roots {
        let value = value.ok_or_else(|| {
            crate::interpreter::errors::mex(
                "RunMat:CommaSeparatedListState",
                "prepared output root is unavailable during publication",
            )
        })?;
        value.with_owned(|value| {
            before_overwrite(slot);
            crate::runtime::workspace::commit_workspace_slot_value(vars, slot, None, value)?;
            after_store(slot);
            authority.publish(vars, slot)?;
            Ok::<_, RuntimeError>(())
        })?;
    }
    stack.extend(retained);
    Ok(())
}
