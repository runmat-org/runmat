use super::Instr;

mod capture;
mod control_flow;
mod prepared;
mod subscript;
#[cfg(test)]
mod tests;
mod transient;
pub(super) use control_flow::normal_successors;

/// Verifies all transient sequence channels and prepared destination state.
pub fn validate_sequence_register_flow(instructions: &[Instr]) -> Result<(), String> {
    control_flow::validate(instructions)?;
    transient::validate(instructions)?;
    capture::validate_sequence_capture_flow(instructions)?;
    prepared::validate(instructions)?;
    subscript::validate_contextual_index_flow(instructions)?;
    subscript::validate_subscript_end_flow(instructions)?;
    subscript::validate_subscript_descriptors(instructions)
}
