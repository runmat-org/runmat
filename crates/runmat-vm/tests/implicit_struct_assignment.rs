#[path = "support/mod.rs"]
mod test_helpers;

#[test]
fn member_assignment_uses_typed_creation_root() {
    let bytecode = test_helpers::compile_source("config.database.port = 5432;").unwrap();
    let (source_statement, root_slot) = bytecode
        .instructions
        .iter()
        .find_map(|instruction| match instruction {
            runmat_vm::Instr::BeginFixedAssignment {
                source_statement,
                root_slot,
            } => Some((*source_statement, *root_slot)),
            _ => None,
        })
        .expect("typed compound member receiver capture");
    assert!(bytecode.instructions.iter().any(|instruction| matches!(instruction,
        runmat_vm::Instr::BeginSequenceOutputTarget { root_slot: actual, root_creation: runmat_mir::SequenceRootCreation::OrdinaryContainer(runmat_types::AssignmentChildCreation::ScalarStructure) }
        if *actual == root_slot)));
    assert!(bytecode
        .instructions
        .iter()
        .any(|instruction| matches!(instruction,
        runmat_vm::Instr::CommitFixedAssignment { source_statement: actual, root_slot: slot }
        if *actual == source_statement && *slot == root_slot)));
}

#[test]
fn missing_formal_input_is_not_initialized_by_member_assignment() {
    let error = test_helpers::execute_source(
        "function y = requires_input(x)\nx.field = 1; y = x;\nend\ny = requires_input();",
    )
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:NotEnoughInputs"));
}
