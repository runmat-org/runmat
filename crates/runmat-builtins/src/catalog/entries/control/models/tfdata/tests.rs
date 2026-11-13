use super::*;

#[test]
fn tfdata_catalog_has_vector_cell_and_indexed_model_forms() {
    assert_eq!(TFDATA_CATALOG_ENTRY.descriptor.signatures.len(), 3);
    assert_eq!(TFDATA_CATALOG_ENTRY.identity.name, "tfdata");
}
