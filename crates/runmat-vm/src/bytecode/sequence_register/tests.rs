mod prepared;
mod subscript;
mod transient;

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
