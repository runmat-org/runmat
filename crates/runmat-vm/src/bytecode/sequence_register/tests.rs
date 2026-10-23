mod prepared;
mod subscript;
mod transient;

#[test]
fn scalar_multi_output_trap_has_only_exceptional_cleanup_not_normal_outputs() {
    use super::Instr;
    let trap = [Instr::Unpack(2), Instr::LoadConst(7.0), Instr::Return];
    assert!(super::normal_successors(&trap, 0).unwrap().is_empty());
    let single = [Instr::Unpack(1), Instr::Return];
    assert_eq!(super::normal_successors(&single, 0), Ok(vec![1]));
    let exceptional = [
        Instr::BeginOutputAssignment { target_count: 1 },
        Instr::PrepareFixedOutputTarget,
        Instr::Unpack(2),
    ];
    assert!(super::validate_sequence_register_flow(&exceptional).is_ok());
    let normal = [
        Instr::BeginOutputAssignment { target_count: 1 },
        Instr::PrepareFixedOutputTarget,
        Instr::Return,
    ];
    assert!(super::validate_sequence_register_flow(&normal).is_err());
    let caught = [
        Instr::EnterTry {
            scope: 0,
            catch_pc: 5,
            catch_var: None,
        },
        Instr::BeginOutputAssignment { target_count: 1 },
        Instr::PrepareFixedOutputTarget,
        Instr::LoadConst(1.),
        Instr::Unpack(2),
        Instr::Return,
    ];
    assert!(super::validate_sequence_register_flow(&caught).is_ok());
    assert_eq!(
        crate::bytecode::derive_stack_geometry(&caught).unwrap(),
        crate::bytecode::BytecodeStackGeometry::Exact { maximum: 1 },
    );
}

#[test]
fn branch_to_the_function_boundary_is_a_valid_exit_edge() {
    let instructions = [super::Instr::Jump(1)];
    assert_eq!(super::normal_successors(&instructions, 0), Ok(vec![1]));
}

#[test]
fn selected_stochastic_source_has_both_original_and_selected_successors() {
    use crate::bytecode::StochasticSourceOperand as Operand;

    let instructions = [
        super::Instr::StochasticEvolution {
            iteration_slot: 0,
            noise_slot: 1,
            state_slot: 2,
            steps: Operand::Number(1.0),
            drift: Operand::Number(0.0),
            scale: Operand::Number(1.0),
            randn_args: Vec::new(),
            fallback_pc: 2,
        },
        super::Instr::Jump(3),
        super::Instr::LoadConst(1.0),
        super::Instr::Return,
    ];
    assert_eq!(super::normal_successors(&instructions, 0), Ok(vec![1, 2]));
    assert_eq!(
        instructions[0].stack_effect(),
        Some(crate::bytecode::StackEffect { pops: 0, pushes: 0 })
    );
    assert!(super::validate_sequence_register_flow(&instructions).is_ok());
}
