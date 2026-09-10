use super::super::super::*;
use runmat_types::MemberName;

fn output_plan(target_count: usize) -> Instr {
    Instr::BeginOutputAssignment { target_count }
}

#[test]
fn accepts_complete_prepared_output_plan_around_rhs() {
    let instructions = [
        output_plan(2),
        Instr::PrepareFixedOutputTarget,
        Instr::PrepareMemberSequenceOutputTarget {
            root_slot: 0,
            member: MemberName::from("field"),
        },
        Instr::LoadPreparedOutputCardinality,
        Instr::LoadVar(1),
        Instr::CaptureScalarSequence,
        Instr::CommitPreparedOutputTargets {
            retained_outputs: 1,
        },
        Instr::Return,
    ];
    assert!(validate_sequence_register_flow(&instructions).is_ok());
}

#[test]
fn rejects_malformed_prepared_output_lifecycle() {
    assert!(
        validate_sequence_register_flow(&[Instr::PrepareFixedOutputTarget, Instr::Return])
            .unwrap_err()
            .contains("without a plan")
    );
    assert!(validate_sequence_register_flow(&[
        output_plan(2),
        Instr::PrepareFixedOutputTarget,
        Instr::LoadPreparedOutputCardinality,
        Instr::Return,
    ])
    .unwrap_err()
    .contains("1/2 targets prepared"));
    assert!(
        validate_sequence_register_flow(&[output_plan(1), output_plan(1), Instr::Return])
            .unwrap_err()
            .contains("overwritten")
    );
    assert!(validate_sequence_register_flow(&[
        output_plan(1),
        Instr::PrepareFixedOutputTarget,
        Instr::LoadPreparedOutputCardinality,
        Instr::Return,
    ])
    .unwrap_err()
    .contains("remains live"));
    assert!(validate_sequence_register_flow(&[
        Instr::CaptureScalarSequence,
        Instr::CommitPreparedOutputTargets {
            retained_outputs: 0,
        },
        Instr::Return,
    ])
    .unwrap_err()
    .contains("without a plan"));
}
