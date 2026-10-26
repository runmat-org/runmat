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
            semantic_fallback: None,
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
