use std::collections::HashMap;

use runmat_runtime::sequence::ResolveValueSequence;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::super::{PreparedOutputTarget, SequenceState};
use super::support::root;

pub(super) async fn commit(
    stack: &mut Vec<Value>,
    vars: &mut [Value],
    state: &mut SequenceState,
    retained_outputs: usize,
    current_function_name: &str,
    before_overwrite: &mut dyn FnMut(&Value, &Value),
    after_store: &mut dyn FnMut(usize, &Value),
) -> Result<(), RuntimeError> {
    let layout = state.output_layout()?;
    let values = super::super::object::take_sequence_register(stack, state)?.resolve(
        runmat_types::SequenceUse::SelectDestinationCardinality,
        runmat_runtime::sequence::SequenceResolutionContext::destination_cardinality(
            layout.total(),
        ),
    )?;
    let partitions = layout.distribute(values)?;
    let targets = state.take_output_targets();
    let mut retained = Vec::with_capacity(retained_outputs);
    let mut roots = Vec::<(usize, Option<Value>)>::new();
    let mut root_positions = HashMap::<usize, usize>::new();
    for (target, values) in targets.into_iter().zip(partitions) {
        match target {
            PreparedOutputTarget::Fixed | PreparedOutputTarget::Discard => retained.extend(values),
            PreparedOutputTarget::Sequence {
                root_slot,
                endpoint,
            } => {
                let position = match root_positions.get(&root_slot).copied() {
                    Some(position) => position,
                    None => {
                        let position = roots.len();
                        roots.push((root_slot, Some(root(vars, root_slot)?.clone())));
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
                    .assign(root_value, values, Some(current_function_name))
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
        before_overwrite(root(vars, slot)?, &value);
        vars[slot] = value;
        after_store(slot, &vars[slot]);
    }
    stack.extend(retained);
    Ok(())
}
