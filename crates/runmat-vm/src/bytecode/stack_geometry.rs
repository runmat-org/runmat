//! Verified operand-stack geometry for one bytecode body.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::{sequence_register::normal_successors, Instr};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BytecodeStackGeometry {
    Exact { maximum: usize },
    Dynamic,
}

impl Default for BytecodeStackGeometry {
    fn default() -> Self {
        Self::Dynamic
    }
}

impl BytecodeStackGeometry {
    pub const fn exact_maximum(self) -> Option<usize> {
        match self {
            Self::Exact { maximum } => Some(maximum),
            Self::Dynamic => None,
        }
    }
}

pub fn derive_stack_geometry(instructions: &[Instr]) -> Result<BytecodeStackGeometry, String> {
    if instructions.is_empty() {
        return Ok(BytecodeStackGeometry::Exact { maximum: 0 });
    }
    let mut entries = vec![None; instructions.len()];
    let mut pending = VecDeque::from([(0usize, 0usize)]);
    let mut maximum = 0usize;
    while let Some((pc, incoming)) = pending.pop_front() {
        if pc == instructions.len() {
            continue;
        }
        if pc > instructions.len() {
            return Err(format!(
                "bytecode operand-stack flow reaches invalid instruction {pc}"
            ));
        }
        if let Some(existing) = entries[pc] {
            if existing != incoming {
                return Err(format!(
                    "bytecode operand-stack depth disagrees at control-flow join instruction {pc}: {existing} versus {incoming}"
                ));
            }
            continue;
        }
        entries[pc] = Some(incoming);
        maximum = maximum.max(incoming);
        let Some(effect) = instructions[pc].stack_effect() else {
            return Ok(BytecodeStackGeometry::Dynamic);
        };
        if incoming < effect.pops {
            return Err(format!(
                "bytecode instruction {pc} pops {} operand(s) from stack depth {incoming}",
                effect.pops
            ));
        }
        let outgoing = incoming
            .checked_sub(effect.pops)
            .and_then(|depth| depth.checked_add(effect.pushes))
            .ok_or_else(|| format!("bytecode operand-stack depth overflows at instruction {pc}"))?;
        maximum = maximum.max(outgoing);
        for successor in normal_successors(instructions, pc)? {
            pending.push_back((successor, outgoing));
        }
        if let Instr::EnterTry { catch_pc, .. } = instructions[pc] {
            if catch_pc > instructions.len() {
                return Err(format!(
                    "bytecode instruction {pc} catches outside the function at {catch_pc}"
                ));
            }
            // Exception transfer restores the operand depth saved before the
            // try body, not the depth at the throwing instruction.
            pending.push_back((catch_pc, incoming));
        }
    }
    Ok(BytecodeStackGeometry::Exact { maximum })
}

pub fn validate_stack_geometry(
    instructions: &[Instr],
    claimed: BytecodeStackGeometry,
) -> Result<(), String> {
    let derived = derive_stack_geometry(instructions)?;
    if derived != claimed {
        return Err(format!(
            "bytecode operand-stack geometry disagrees with its instruction stream: claimed {claimed:?}, derived {derived:?}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_exact_branch_join_capacity() {
        let instructions = vec![
            Instr::LoadBool(true),
            Instr::JumpIfFalse(4),
            Instr::LoadInt(runmat_value::IntValue::I64(1)),
            Instr::Jump(5),
            Instr::LoadInt(runmat_value::IntValue::I64(2)),
            Instr::ReturnValue,
        ];
        assert_eq!(
            derive_stack_geometry(&instructions),
            Ok(BytecodeStackGeometry::Exact { maximum: 1 })
        );
    }

    #[test]
    fn rejects_inconsistent_join_depth() {
        let instructions = vec![
            Instr::LoadBool(true),
            Instr::JumpIfFalse(4),
            Instr::LoadInt(runmat_value::IntValue::I64(1)),
            Instr::Jump(5),
            Instr::Jump(5),
            Instr::Return,
        ];
        assert!(derive_stack_geometry(&instructions)
            .unwrap_err()
            .contains("disagrees at control-flow join"));
    }
}
