use super::*;
use std::collections::BTreeMap;

use runmat_types::{
    NumericClass, NumericDomain, NumericFact, ObjectFact, ResidencyFact, ShapeFact, StorageFact,
    ValueKindFact,
};

#[test]
fn array_counts_preserve_label_class_and_type_count_outputs_as_double_columns() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(8), Some(1)]),
        StorageFact::Dense,
    );
    let inferred = infer("groupcounts", vec![input], RequestedOutputCount::Exactly(3));
    assert_eq!(inferred.outputs.len(), 3);
    assert_eq!(
        inferred.outputs[0].numeric().map(|numeric| numeric.class),
        Some(NumericClass::Double)
    );
    assert_eq!(
        inferred.outputs[1].numeric().map(|numeric| numeric.class),
        Some(NumericClass::UInt64)
    );
    assert_eq!(
        inferred.outputs[1].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    assert_eq!(inferred.outputs[2].residency, ResidencyFact::Host);
}

#[test]
fn tabular_counts_return_a_host_table_fact() {
    let table = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::TIMETABLE.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let inferred = infer("groupcounts", vec![table], RequestedOutputCount::One);
    assert_eq!(inferred.outputs.len(), 1);
    assert!(
        matches!(&inferred.outputs[0].kind, ValueKindFact::Object(object) if object.runtime_class.as_ref().is_some_and(|class| class.is(runmat_types::standard::TABLE)))
    );
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn matrix_group_labels_are_a_cell_with_one_entry_per_column() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int32,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(8), Some(3)]),
        StorageFact::Dense,
    );
    let inferred = infer("groupcounts", vec![input], RequestedOutputCount::Exactly(2));
    assert_eq!(
        inferred.outputs[1].shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );
    assert!(matches!(inferred.outputs[1].kind, ValueKindFact::Cell(_)));
}
