use super::*;
use std::collections::BTreeMap;

use runmat_types::{
    NumericClass, NumericDomain, NumericFact, ObjectFact, ResidencyFact, ShapeFact, StorageFact,
    ValueKindFact,
};

#[test]
fn vector_groups_preserve_orientation_and_identifier_class() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let inferred = infer("findgroups", vec![input], RequestedOutputCount::Exactly(2));
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );
    assert_eq!(
        inferred.outputs[0].numeric().map(|numeric| numeric.class),
        Some(NumericClass::Double)
    );
    assert_eq!(
        inferred.outputs[1].numeric().map(|numeric| numeric.class),
        Some(NumericClass::UInt64)
    );
    assert_eq!(inferred.outputs[1].residency, ResidencyFact::Host);
}

#[test]
fn table_form_types_one_identifier_table_and_rejects_excess_outputs() {
    let table = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::TABLE.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let inferred = infer("findgroups", vec![table], RequestedOutputCount::Exactly(3));
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FINDGROUPS-OUTPUTS"));
}

#[test]
fn matrix_extension_types_one_identifier_output_per_column() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(5), Some(2)]),
        StorageFact::Dense,
    );
    let inferred = infer("findgroups", vec![input], RequestedOutputCount::Exactly(3));
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(5), Some(1)])
    );
    assert_eq!(inferred.outputs.len(), 3);
    for output in &inferred.outputs[1..] {
        assert_eq!(
            output.numeric().map(|numeric| numeric.class),
            Some(NumericClass::Single)
        );
    }
}
