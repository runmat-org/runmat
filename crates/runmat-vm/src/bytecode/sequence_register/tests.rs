mod prepared;
mod subscript;
mod transient;

#[test]
fn branch_to_the_function_boundary_is_a_valid_exit_edge() {
    let instructions = [super::Instr::Jump(1)];
    assert_eq!(super::normal_successors(&instructions, 0), Ok(vec![1]));
}
