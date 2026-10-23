//! Branch admission and successor enumeration shared by sequence verifiers.

use super::Instr;

pub(super) fn validate(instructions: &[Instr]) -> Result<(), String> {
    for (pc, instruction) in instructions.iter().enumerate() {
        match instruction {
            Instr::Jump(target)
            | Instr::JumpIfFalse(target)
            | Instr::AndAnd(target)
            | Instr::OrOr(target) => {
                checked_target(instructions.len(), pc, *target)?;
            }
            Instr::StochasticEvolution { fallback_pc, .. } => {
                checked_stochastic_fallback(instructions.len(), pc, *fallback_pc)?;
            }
            _ => {}
        }
    }
    Ok(())
}

pub(in crate::bytecode) fn normal_successors(
    instructions: &[Instr],
    pc: usize,
) -> Result<Vec<usize>, String> {
    match &instructions[pc] {
        instruction if is_exceptional_exit(instruction) => Ok(Vec::new()),
        Instr::Return | Instr::ReturnValue => Ok(Vec::new()),
        Instr::Jump(target) => Ok(vec![checked_target(instructions.len(), pc, *target)?]),
        Instr::JumpIfFalse(target) | Instr::AndAnd(target) | Instr::OrOr(target) => {
            let mut successors = vec![checked_target(instructions.len(), pc, *target)?];
            if pc + 1 < instructions.len() {
                successors.push(pc + 1);
            }
            Ok(successors)
        }
        Instr::StochasticEvolution { fallback_pc, .. } => {
            let target = checked_stochastic_fallback(instructions.len(), pc, *fallback_pc)?;
            Ok(vec![pc + 1, target])
        }
        _ if pc + 1 < instructions.len() => Ok(vec![pc + 1]),
        _ => Ok(Vec::new()),
    }
}

/// This instruction never returns normally. Error routing clears the same
/// sequence/prepared owners on catch, or drops them with the failed frame.
pub(super) fn is_exceptional_exit(instruction: &Instr) -> bool {
    matches!(instruction, Instr::Unpack(count) if *count > 1)
}

fn checked_target(len: usize, pc: usize, target: usize) -> Result<usize, String> {
    (target <= len).then_some(target).ok_or_else(|| {
        format!("bytecode instruction {pc} branches outside the function to {target}")
    })
}

fn checked_stochastic_fallback(len: usize, pc: usize, target: usize) -> Result<usize, String> {
    checked_target(len, pc, target)?;
    if target >= len || target <= pc + 1 {
        return Err(format!(
            "stochastic source instruction {pc} lacks a following original-loop fallback at {target}"
        ));
    }
    Ok(target)
}
