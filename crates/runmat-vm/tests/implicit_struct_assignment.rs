#[path = "support/mod.rs"]
mod test_helpers;

#[test]
fn member_assignment_uses_typed_creation_root() {
    let bytecode = test_helpers::compile_source("config.database.port = 5432;").unwrap();
    assert!(bytecode
        .instructions
        .iter()
        .any(|instruction| matches!(instruction, runmat_vm::Instr::LoadVarForStructAssignment(_))));
}

#[test]
fn missing_formal_input_is_not_initialized_by_member_assignment() {
    let error = test_helpers::execute_source(
        "function y = requires_input(x)\nx.field = 1; y = x;\nend\ny = requires_input();",
    )
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:NotEnoughInputs"));
}
