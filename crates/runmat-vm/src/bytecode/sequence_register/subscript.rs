use std::collections::{BTreeMap, VecDeque};

use super::{normal_successors, Instr};

pub(super) fn validate_subscript_descriptors(instructions: &[Instr]) -> Result<(), String> {
    for (pc, instruction) in instructions.iter().enumerate() {
        let (steps, selection, to_register) = match instruction {
            Instr::ReadSubscriptPath {
                steps,
                selection,
                to_sequence_register,
                ..
            } => (steps, Some(*selection), *to_sequence_register),
            Instr::CaptureSubscriptPath { steps, .. } => (steps, None, false),
            Instr::BeginSubscriptEndReceiver { prefix } => (prefix, None, false),
            _ => continue,
        };
        if !matches!(instruction, Instr::BeginSubscriptEndReceiver { .. }) && steps.is_empty() {
            return Err(format!("subscript path at instruction {pc} is empty"));
        }
        steps.iter().try_fold(0usize, |total, step| {
            total.checked_add(step.operand_count()).ok_or_else(|| {
                format!("subscript path operand count overflows at instruction {pc}")
            })
        })?;
        if to_register && selection != Some(runmat_types::SequenceUse::SelectDestinationCardinality)
        {
            return Err(format!(
                "subscript path at instruction {pc} writes the sequence register without destination-cardinality selection"
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_subscript_end_flow(instructions: &[Instr]) -> Result<(), String> {
    validate_nested_flow(instructions, SubscriptFlow::End)
}

pub(super) fn validate_contextual_index_flow(instructions: &[Instr]) -> Result<(), String> {
    validate_nested_flow(instructions, SubscriptFlow::ContextualIndex)
}

#[derive(Clone, Copy)]
enum SubscriptFlow {
    End,
    ContextualIndex,
}

fn validate_nested_flow(instructions: &[Instr], flow: SubscriptFlow) -> Result<(), String> {
    if instructions.is_empty() {
        return Ok(());
    }
    let label = match flow {
        SubscriptFlow::End => "subscript end receiver",
        SubscriptFlow::ContextualIndex => "contextual index state",
    };
    let mut entries = BTreeMap::<usize, Vec<usize>>::new();
    let mut pending = VecDeque::from([(0usize, Vec::new())]);
    while let Some((pc, incoming)) = pending.pop_front() {
        if pc >= instructions.len() {
            if incoming.is_empty() {
                continue;
            }
            return Err(format!("{label} remains live at the function boundary"));
        }
        if let Some(existing) = entries.get(&pc) {
            if existing != &incoming {
                return Err(format!(
                    "{label} disagrees at control-flow join instruction {pc}"
                ));
            }
            continue;
        }
        entries.insert(pc, incoming.clone());
        let mut outgoing = incoming;
        match flow {
            SubscriptFlow::End => update_end(&instructions[pc], pc, &mut outgoing)?,
            SubscriptFlow::ContextualIndex => {
                update_contextual(&instructions[pc], pc, &mut outgoing)?
            }
        }
        let successors = normal_successors(instructions, pc)?;
        if successors.is_empty() && !outgoing.is_empty() {
            return Err(format!(
                "{label} remains live at function exit instruction {pc}"
            ));
        }
        pending.extend(successors.into_iter().map(|next| (next, outgoing.clone())));
    }
    Ok(())
}

fn update_end(instruction: &Instr, pc: usize, state: &mut Vec<usize>) -> Result<(), String> {
    match instruction {
        Instr::BeginSubscriptEndReceiver { prefix } => {
            let operands = prefix.iter().try_fold(0usize, |total, step| {
                total.checked_add(step.operand_count()).ok_or_else(|| {
                    format!("subscript prefix operand count overflows at instruction {pc}")
                })
            })?;
            state.push(operands);
        }
        Instr::LoadSubscriptEnd {
            component,
            component_count,
        } => {
            if state.is_empty() {
                return Err(format!(
                    "subscript end at instruction {pc} has no prepared receiver"
                ));
            }
            if component >= component_count {
                return Err(format!("subscript end component {component} exceeds selector count {component_count} at instruction {pc}"));
            }
        }
        Instr::FinishSubscriptEndReceiver {
            selector_count,
            prefix_operand_count,
        } => {
            let Some(expected_prefix_operands) = state.pop() else {
                return Err(format!(
                    "subscript end receiver finishes at instruction {pc} without a live receiver"
                ));
            };
            if expected_prefix_operands != *prefix_operand_count {
                return Err(format!(
                    "subscript end receiver at instruction {pc} declares {prefix_operand_count} prefix operands but prepared {expected_prefix_operands}"
                ));
            }
            if *selector_count == 0 {
                return Err(format!(
                    "subscript end receiver at instruction {pc} has no selector components"
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

fn update_contextual(instruction: &Instr, pc: usize, state: &mut Vec<usize>) -> Result<(), String> {
    match instruction {
        Instr::BeginContextualIndexSelectors { component_count } => {
            if *component_count == 0 {
                return Err(format!(
                    "contextual index at instruction {pc} has no selector components"
                ));
            }
            state.push(*component_count);
        }
        Instr::LoadContextualIndexEnd { component } => {
            let count = state.last().ok_or_else(|| {
                format!("contextual end at instruction {pc} has no live index context")
            })?;
            if component >= count {
                return Err(format!("contextual end component {component} exceeds selector count {count} at instruction {pc}"));
            }
        }
        Instr::FinishContextualIndexSelectors { component_count } => {
            let count = state.pop().ok_or_else(|| {
                format!("contextual index finishes at instruction {pc} without a live context")
            })?;
            if count != *component_count {
                return Err(format!("contextual index at instruction {pc} finishes with {component_count} components but began with {count}"));
            }
        }
        _ => {}
    }
    Ok(())
}
