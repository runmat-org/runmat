use super::super::inference::test_support::{numeric, request};
use super::MINUS_CATALOG_ENTRY;
use crate::infer_catalog_call;
use runmat_types::{NumericClass, NumericDomain};

#[test]
fn rejects_mixed_integer_classes_without_losing_broadcast_diagnostics() {
    let inferred = infer_catalog_call(
        &MINUS_CATALOG_ENTRY,
        &request(vec![
            numeric(
                NumericClass::Int16,
                NumericDomain::Real,
                &[Some(2), Some(2)],
            ),
            numeric(
                NumericClass::UInt16,
                NumericDomain::Real,
                &[Some(3), Some(2)],
            ),
        ]),
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-BINARY-ARITHMETIC-INPUT"));
    assert_eq!(inferred.diagnostics.len(), 2);
}
