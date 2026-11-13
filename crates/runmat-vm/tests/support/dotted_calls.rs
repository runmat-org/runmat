use runmat_runtime::call::arguments::ArgumentSpec;
use runmat_types::MemberName;
use runmat_vm::{BytecodeDottedInvokeArgument, BytecodeSubscriptStep, Instr};

/// Check the typed member, input cardinality and output selection together.
/// These fixtures supply individual values, not expanding sequences.
pub(super) fn has_single_value_dotted_call(
    instructions: &[Instr],
    expected_member: &MemberName,
    expected_inputs: usize,
    expected_outputs: usize,
) -> bool {
    instructions.iter().any(|instruction| {
        let Instr::ReadSubscriptPath {
            steps, selection, ..
        } = instruction
        else {
            return false;
        };
        let [BytecodeSubscriptStep::DottedInvoke {
            member, arguments, ..
        }] = steps.as_slice()
        else {
            return false;
        };
        member == expected_member
            && selection.output_count() == Some(expected_outputs)
            && arguments.len() == expected_inputs
            && arguments.iter().all(|argument| {
                matches!(
                    argument,
                    BytecodeDottedInvokeArgument::Call(ArgumentSpec::Single)
                )
            })
    })
}
