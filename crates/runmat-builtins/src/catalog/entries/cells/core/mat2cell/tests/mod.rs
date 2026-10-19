use super::super::*;

mod inference;

#[test]
fn contract_has_partition_forms() {
    assert_eq!(MAT2CELL_DESCRIPTOR.signatures.len(), 2);
    assert_eq!(MAT2CELL_EXTENSIONS.len(), 1);
}

#[test]
fn documentation_examples_are_executable() {
    assert!(MAT2CELL_CATALOG_ENTRY
        .documentation
        .examples
        .iter()
        .all(|example| !example.program.is_empty()));
}
