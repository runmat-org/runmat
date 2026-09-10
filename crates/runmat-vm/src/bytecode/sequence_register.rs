use super::Instr;

mod capture;
mod prepared;
mod subscript;
#[cfg(test)]
mod tests;
mod transient;

/// Verifies all transient sequence channels and prepared destination state.
pub fn validate_sequence_register_flow(instructions: &[Instr]) -> Result<(), String> {
    transient::validate(instructions)?;
    capture::validate_sequence_capture_flow(instructions)?;
    prepared::validate(instructions)?;
    subscript::validate_contextual_index_flow(instructions)?;
    subscript::validate_subscript_end_flow(instructions)?;
    subscript::validate_subscript_descriptors(instructions)
}

pub(super) fn normal_successors(instructions: &[Instr], pc: usize) -> Result<Vec<usize>, String> {
    let checked_target = |target: usize| {
        (target < instructions.len())
            .then_some(target)
            .ok_or_else(|| {
                format!("bytecode instruction {pc} branches outside the function to {target}")
            })
    };
    match &instructions[pc] {
        Instr::Return | Instr::ReturnValue => Ok(Vec::new()),
        Instr::Jump(target) => Ok(vec![checked_target(*target)?]),
        Instr::JumpIfFalse(target) | Instr::AndAnd(target) | Instr::OrOr(target) => {
            let mut successors = vec![checked_target(*target)?];
            if pc + 1 < instructions.len() {
                successors.push(pc + 1);
            }
            Ok(successors)
        }
        _ if pc + 1 < instructions.len() => Ok(vec![pc + 1]),
        _ => Ok(Vec::new()),
    }
}
