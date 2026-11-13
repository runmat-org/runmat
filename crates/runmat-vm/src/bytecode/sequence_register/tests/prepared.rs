use super::super::super::*;
use runmat_types::MemberName;

fn output_plan(target_count: usize) -> Instr {
    Instr::BeginOutputAssignment { target_count }
}

#[test]
fn fixed_preparation_binds_source_root_and_cannot_reenter_through_catch() {
    let source = runmat_types::StmtId(7);
    let plan = vec![
        Instr::BeginFixedAssignment {
            source_statement: source,
            root_slot: 0,
        },
        Instr::BeginSequenceOutputTarget {
            root_slot: 0,
            root_creation: runmat_mir::SequenceRootCreation::Ordinary(
                runmat_types::AssignmentCreationPolicy::CreateStructFieldPath,
            ),
        },
        Instr::FinishFixedAssignmentTarget {
            endpoint: crate::bytecode::BytecodeSubscriptStep::Member(MemberName::from("value")),
            selector_context: None,
            mutation: runmat_types::PlaceMutationKind::MemberAssign,
        },
        Instr::LoadPreparedOutputCardinality,
        Instr::Pop,
        Instr::LoadVar(1),
        Instr::CaptureScalarSequence,
        Instr::CommitFixedAssignment {
            source_statement: source,
            root_slot: 0,
        },
        Instr::Return,
    ];
    validate_sequence_register_flow(&plan).unwrap();
    let mut wrong_root = plan.clone();
    if let Instr::BeginSequenceOutputTarget { root_slot, .. } = &mut wrong_root[1] {
        *root_slot = 2;
    }
    assert!(validate_sequence_register_flow(&wrong_root)
        .unwrap_err()
        .contains("root disagrees"));
    let mut wrong_source = plan.clone();
    if let Instr::CommitFixedAssignment {
        source_statement, ..
    } = &mut wrong_source[7]
    {
        source_statement.0 += 1;
    }
    assert!(validate_sequence_register_flow(&wrong_source).is_err());
    let mut catch_without_owner = vec![Instr::EnterTry {
        scope: 0,
        catch_pc: 8,
        catch_var: None,
    }];
    catch_without_owner.extend(plan);
    assert!(validate_sequence_register_flow(&catch_without_owner)
        .unwrap_err()
        .contains("without preparation"));
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
