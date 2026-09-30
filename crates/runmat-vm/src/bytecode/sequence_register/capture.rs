use std::collections::{BTreeMap, BTreeSet, VecDeque};

use runmat_runtime::call::arguments::ArgumentSpec;

use super::{normal_successors, Instr};

pub(super) fn validate_sequence_capture_flow(instructions: &[Instr]) -> Result<(), String> {
    if instructions.is_empty() {
        return Ok(());
    }
    let mut entry_states = BTreeMap::<usize, BTreeSet<usize>>::new();
    let mut pending = VecDeque::from([(0usize, BTreeSet::new())]);
    while let Some((pc, incoming)) = pending.pop_front() {
        if pc >= instructions.len() {
            if incoming.is_empty() {
                continue;
            }
            return Err(format!("captured comma-separated sequence locals {incoming:?} remain live at the function boundary"));
        }
        if let Some(existing) = entry_states.get(&pc) {
            if existing != &incoming {
                return Err(format!("captured comma-separated sequence state disagrees at control-flow join instruction {pc}: {existing:?} versus {incoming:?}"));
            }
            continue;
        }
        entry_states.insert(pc, incoming.clone());
        let mut outgoing = incoming;
        if let Some(slot) = capture_slot(&instructions[pc]) {
            if !outgoing.insert(slot) {
                return Err(format!("captured comma-separated sequence local {slot} is overwritten at instruction {pc} before consumption"));
            }
        }
        let mut consumed = BTreeSet::new();
        if let Some(specs) = captured_argument_specs(&instructions[pc]) {
            for slot in specs.iter().filter_map(|spec| match spec {
                ArgumentSpec::CapturedSequence { slot } => Some(*slot),
                _ => None,
            }) {
                consume(slot, pc, &mut consumed, &mut outgoing)?;
            }
        }
        for slot in captured_aggregate_slots(&instructions[pc]) {
            consume(slot, pc, &mut consumed, &mut outgoing)?;
        }
        for slot in captured_dotted_argument_slots(&instructions[pc]) {
            consume(slot, pc, &mut consumed, &mut outgoing)?;
        }
        let successors = normal_successors(instructions, pc)?;
        if successors.is_empty() && !outgoing.is_empty() {
            return Err(format!("captured comma-separated sequence locals {outgoing:?} remain live at function exit instruction {pc}"));
        }
        pending.extend(successors.into_iter().map(|next| (next, outgoing.clone())));
    }
    Ok(())
}

fn consume(
    slot: usize,
    pc: usize,
    consumed: &mut BTreeSet<usize>,
    live: &mut BTreeSet<usize>,
) -> Result<(), String> {
    if !consumed.insert(slot) {
        return Err(format!("captured comma-separated sequence local {slot} is consumed more than once by instruction {pc}"));
    }
    if !live.remove(&slot) {
        return Err(format!("captured comma-separated sequence local {slot} is consumed at instruction {pc} without a live capture"));
    }
    Ok(())
}

fn capture_slot(instruction: &Instr) -> Option<usize> {
    match instruction {
        Instr::CaptureMemberSequence { sequence_slot, .. }
        | Instr::CaptureMemberDynamicSequence { sequence_slot }
        | Instr::CaptureCellContentsSequence { sequence_slot, .. }
        | Instr::CaptureReturnedOutputsSequence { sequence_slot }
        | Instr::CaptureSubscriptPath { sequence_slot, .. } => Some(*sequence_slot),
        _ => None,
    }
}

fn captured_argument_specs(instruction: &Instr) -> Option<&[ArgumentSpec]> {
    match instruction {
        Instr::CallFevalExpandMultiOutput(specs, _)
        | Instr::CallFevalExpandMultiOutputUsingOutputSlot(specs, _)
        | Instr::CreateSemanticFutureExpandMultiOutput(_, specs, _)
        | Instr::CallSemanticFunctionExpandMultiOutput(_, specs, _)
        | Instr::CallBuiltinExpandMultiOutput(_, specs, _)
        | Instr::CallFunctionExpandMultiOutput { specs, .. }
        | Instr::CallWorkspaceFirstExpandMultiOutput { specs, .. }
        | Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot { specs, .. }
        | Instr::CallSemanticNestedFunctionExpandMultiOutput { specs, .. }
        | Instr::CallSuperConstructorExpandMultiOutput { specs, .. }
        | Instr::CallSuperMethodExpandMultiOutput { specs, .. }
        | Instr::CallMethodOrMemberIndexExpandMultiOutput { specs, .. } => Some(specs),
        _ => None,
    }
}

fn captured_aggregate_slots(instruction: &Instr) -> impl Iterator<Item = usize> + '_ {
    match instruction {
        Instr::CreateMatrixFromSequences { elements, .. }
        | Instr::CreateCellFromSequences { elements, .. } => Some(elements.as_slice()),
        _ => None,
    }
    .into_iter()
    .flatten()
    .filter_map(|element| match element {
        crate::bytecode::AggregateElementSpec::CapturedSequence { slot } => Some(*slot),
        crate::bytecode::AggregateElementSpec::Single => None,
    })
}

fn captured_dotted_argument_slots(instruction: &Instr) -> impl Iterator<Item = usize> + '_ {
    let steps = match instruction {
        Instr::ReadSubscriptPath { steps, .. } | Instr::CaptureSubscriptPath { steps, .. } => {
            Some(steps.as_slice())
        }
        _ => None,
    };
    steps
        .into_iter()
        .flatten()
        .flat_map(|step| match step {
            crate::bytecode::BytecodeSubscriptStep::DottedInvoke { arguments, .. } => {
                Some(arguments.as_slice())
            }
            _ => None,
        })
        .flatten()
        .filter_map(|argument| match argument {
            crate::bytecode::BytecodeDottedInvokeArgument::Call(
                ArgumentSpec::CapturedSequence { slot },
            ) => Some(*slot),
            crate::bytecode::BytecodeDottedInvokeArgument::Colon
            | crate::bytecode::BytecodeDottedInvokeArgument::Call(_) => None,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::{BytecodeDottedInvokeArgument, BytecodeSubscriptStep};
    use runmat_types::{MemberName, ObjectIndexingContext, SequenceUse};

    fn capture(slot: usize) -> Instr {
        Instr::CaptureMemberSequence {
            member: MemberName::from("values"),
            sequence_slot: slot,
        }
    }

    fn invoke(slots: &[usize]) -> Instr {
        Instr::ReadSubscriptPath {
            steps: vec![BytecodeSubscriptStep::DottedInvoke {
                member: MemberName::from("method"),
                arguments: slots
                    .iter()
                    .map(|slot| {
                        BytecodeDottedInvokeArgument::Call(ArgumentSpec::CapturedSequence {
                            slot: *slot,
                        })
                    })
                    .collect(),
            }],
            selection: SequenceUse::RequireSingle,
            context: ObjectIndexingContext::Expression,
            to_sequence_register: false,
        }
    }

    #[test]
    fn dotted_argument_consumes_each_captured_sequence_once() {
        assert!(validate_sequence_capture_flow(&[
            capture(1),
            capture(2),
            invoke(&[1, 2]),
            Instr::Return
        ])
        .is_ok());
        assert!(
            validate_sequence_capture_flow(&[invoke(&[1]), Instr::Return])
                .unwrap_err()
                .contains("without a live capture")
        );
        assert!(
            validate_sequence_capture_flow(&[capture(1), invoke(&[1, 1]), Instr::Return])
                .unwrap_err()
                .contains("more than once")
        );
    }
}
