use runmat_runtime::call::arguments::ArgumentSpec;
use runmat_vm::{
    BytecodeDottedInvokeArgument, BytecodeSubscriptSelector, BytecodeSubscriptStep, Instr,
};

fn latest_sequence_producer<'a>(prefix: &'a [Instr], spec: &ArgumentSpec) -> Option<&'a Instr> {
    let ArgumentSpec::CapturedSequence { slot } = spec else {
        return None;
    };
    prefix.iter().rev().find(|instruction| match instruction {
        Instr::CaptureCellContentsSequence { sequence_slot, .. }
        | Instr::CaptureMemberSequence { sequence_slot, .. }
        | Instr::CaptureMemberDynamicSequence { sequence_slot }
        | Instr::CaptureReturnedOutputsSequence { sequence_slot }
        | Instr::CaptureSubscriptPath { sequence_slot, .. } => sequence_slot == slot,
        _ => false,
    })
}

/// Match the call's captured argument to its preceding cell-content producer.
/// These fixtures have straight-line captures; call identity, arity, fallback
/// policy, and execution assertions remain at each owning test.
pub(super) fn is_expand_all_cell_contents(
    prefix: &[Instr],
    spec: &ArgumentSpec,
    expected_num_indices: usize,
) -> bool {
    let producer = latest_sequence_producer(prefix, spec);
    matches!(producer, Some(Instr::CaptureCellContentsSequence {
        num_indices, expand_all: true, ..
    }) if *num_indices == expected_num_indices)
}

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

#[test]
fn captured_cell_witness_requires_matching_producer_and_selector_contract() {
    let producer = Instr::CaptureCellContentsSequence {
        sequence_slot: 3,
        num_indices: 0,
        expand_all: true,
    };
    let spec = ArgumentSpec::CapturedSequence { slot: 3 };
    assert!(is_expand_all_cell_contents(
        std::slice::from_ref(&producer),
        &spec,
        0
    ));
    assert!(!is_expand_all_cell_contents(&[], &spec, 0));
    assert!(!is_expand_all_cell_contents(
        std::slice::from_ref(&producer),
        &ArgumentSpec::Single,
        0,
    ));
    assert!(!is_expand_all_cell_contents(
        &[Instr::CaptureCellContentsSequence {
            sequence_slot: 3,
            num_indices: 0,
            expand_all: false,
        }],
        &spec,
        0,
    ));
    assert!(!is_expand_all_cell_contents(
        std::slice::from_ref(&producer),
        &spec,
        1
    ));
    assert!(!is_expand_all_cell_contents(
        std::slice::from_ref(&producer),
        &ArgumentSpec::CapturedSequence { slot: 4 },
        0,
    ));
    assert!(!is_expand_all_cell_contents(
        &[
            producer,
            Instr::CaptureReturnedOutputsSequence { sequence_slot: 3 }
        ],
        &spec,
        0,
    ));
}
