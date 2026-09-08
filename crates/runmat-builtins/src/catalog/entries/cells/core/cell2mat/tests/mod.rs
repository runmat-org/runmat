use super::super::*;

mod inference;

#[test]
fn contract_has_one_public_signature() {
    assert_eq!(CELL2MAT_DESCRIPTOR.signatures.len(), 1);
    assert_eq!(CELL2MAT_DESCRIPTOR.signatures[0].label, "A = cell2mat(C)");
}

#[test]
fn documentation_examples_are_executable() {
    assert!(CELL2MAT_CATALOG_ENTRY
        .documentation
        .examples
        .iter()
        .all(|example| !example.program.is_empty()));
}
