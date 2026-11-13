use super::captured_arguments::latest_sequence_producer;
use runmat_vm::{
    BytecodeDottedInvokeArgument, BytecodeSubscriptSelector, BytecodeSubscriptStep, Instr,
};

/// The dotted call must consume a sequence captured from an all-colon cell
/// selector, not an unrelated producer that happens to reuse the same slot.
pub(super) fn has_captured_cell_dotted_call(
    instructions: &[Instr],
    expected_num_outputs: usize,
) -> bool {
    instructions
        .iter()
        .enumerate()
        .any(|(call_position, instruction)| {
            let steps = match instruction {
                Instr::ReadSubscriptPath {
                    steps, selection, ..
                } if selection.output_count() == Some(expected_num_outputs) => steps,
                _ => return false,
            };
            let [BytecodeSubscriptStep::DottedInvoke { arguments, .. }] = steps.as_slice() else {
                return false;
            };
            let [BytecodeDottedInvokeArgument::Call(spec)] = arguments.as_slice() else {
                return false;
            };
            match latest_sequence_producer(&instructions[..call_position], spec) {
                Some(Instr::CaptureSubscriptPath { steps, .. }) => matches!(
                    steps.as_slice(),
                    [BytecodeSubscriptStep::Braces { selectors }]
                        if matches!(selectors.as_slice(), [BytecodeSubscriptSelector::Colon])
                ),
                Some(Instr::CaptureCellContentsSequence {
                    num_indices: 0,
                    expand_all: true,
                    ..
                }) => true,
                _ => false,
            }
        })
}
