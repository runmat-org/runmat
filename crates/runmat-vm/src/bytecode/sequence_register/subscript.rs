mod descriptor;
mod flow;

use super::Instr;

pub(super) fn validate_subscript_descriptors(instructions: &[Instr]) -> Result<(), String> {
    descriptor::validate_subscript_descriptors(instructions)
}

pub(super) fn validate_subscript_end_flow(instructions: &[Instr]) -> Result<(), String> {
    flow::validate_subscript_end_flow(instructions)
}

pub(super) fn validate_contextual_index_flow(instructions: &[Instr]) -> Result<(), String> {
    flow::validate_contextual_index_flow(instructions)
}
