use super::super::super::*;
use runmat_runtime::call::arguments::ArgumentSpec;
use runmat_types::MemberName;

fn producer() -> Instr {
    Instr::LoadMemberSequenceUsingOutputSlot {
        member: MemberName::from("field"),
        output_count_slot: 0,
    }
}

fn captured_call(slot: usize) -> Instr {
    Instr::CallBuiltinExpandMultiOutput(
        "sink".into(),
        vec![ArgumentSpec::CapturedSequence { slot }],
        1,
    )
}

#[test]
fn accepts_adjacent_producer_and_consumer() {
    assert!(validate_sequence_register_flow(&[
        producer(),
        Instr::StoreMemberSequence(MemberName::from("field")),
    ])
    .is_ok());
}

#[test]
fn rejects_overwrite_empty_read_intervening_instruction_and_leak() {
    assert!(validate_sequence_register_flow(&[producer(), producer()])
        .unwrap_err()
        .contains("overwritten"));
    assert!(
        validate_sequence_register_flow(&[Instr::StoreMemberDynamicSequence])
            .unwrap_err()
            .contains("without a producer")
    );
    for instruction in [Instr::LoadConst(1.0), Instr::Jump(0)] {
        assert!(validate_sequence_register_flow(&[producer(), instruction])
            .unwrap_err()
            .contains("crosses instruction"));
    }
    assert!(validate_sequence_register_flow(&[producer()])
        .unwrap_err()
        .contains("function boundary"));
}

#[test]
fn legacy_capture_requires_an_immediate_dynamic_call() {
    assert!(validate_sequence_register_flow(&[
        Instr::LoadConst(1.0),
        Instr::CaptureCallOutputSequence,
        Instr::StoreMemberSequence(MemberName::from("field")),
    ])
    .unwrap_err()
    .contains("immediately follow"));
}

#[test]
fn captured_sequence_may_cross_matching_control_flow() {
    let instructions = [
        Instr::LoadVar(0),
        Instr::CaptureMemberSequence {
            member: MemberName::from("field"),
            sequence_slot: 4,
        },
        Instr::LoadBool(true),
        Instr::JumpIfFalse(5),
        Instr::Jump(5),
        captured_call(4),
        Instr::Return,
    ];
    assert!(validate_sequence_register_flow(&instructions).is_ok());
}

#[test]
fn rejects_capture_overwrite_missing_duplicate_and_leak() {
    let capture = || Instr::CaptureMemberSequence {
        member: MemberName::from("field"),
        sequence_slot: 2,
    };
    assert!(
        validate_sequence_register_flow(&[capture(), capture(), Instr::Return])
            .unwrap_err()
            .contains("overwritten")
    );
    assert!(
        validate_sequence_register_flow(&[captured_call(2), Instr::Return])
            .unwrap_err()
            .contains("without a live capture")
    );
    let duplicate = Instr::CallBuiltinExpandMultiOutput(
        "sink".into(),
        vec![
            ArgumentSpec::CapturedSequence { slot: 2 },
            ArgumentSpec::CapturedSequence { slot: 2 },
        ],
        1,
    );
    assert!(
        validate_sequence_register_flow(&[capture(), duplicate, Instr::Return])
            .unwrap_err()
            .contains("more than once")
    );
    assert!(validate_sequence_register_flow(&[capture(), Instr::Return])
        .unwrap_err()
        .contains("remain live"));
}

#[test]
fn rejects_capture_state_disagreement_at_join() {
    let instructions = [
        Instr::LoadBool(true),
        Instr::JumpIfFalse(4),
        Instr::LoadVar(0),
        Instr::CaptureMemberSequence {
            member: MemberName::from("field"),
            sequence_slot: 1,
        },
        Instr::LoadConst(0.0),
        Instr::Return,
    ];
    assert!(validate_sequence_register_flow(&instructions)
        .unwrap_err()
        .contains("disagrees at control-flow join"));
}
