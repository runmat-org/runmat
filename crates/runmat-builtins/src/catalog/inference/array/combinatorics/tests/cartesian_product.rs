use super::*;
use crate::{
    builtin_catalog_entry_by_name, ArrayInferenceRule, BuiltinInferenceRule,
    CombinatoricsInferenceRule,
};
use runmat_types::{ObjectFact, ResidencyFact, StructFact};

#[test]
fn infers_typed_table_variables_and_cartesian_height() {
    let entry = builtin_catalog_entry_by_name("combinations").expect("combinations entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Array(ArrayInferenceRule::Combinatorics(
            CombinatoricsInferenceRule::CartesianProduct,
        ))
    );
    let inferred = crate::infer_catalog_call(
        entry,
        &request(vec![
            numeric(
                NumericClass::UInt64,
                ShapeFact::from(vec![Some(1), Some(2)]),
            ),
            ValueFact::proven(
                ValueKindFact::String,
                ShapeFact::from(vec![Some(1), Some(3)]),
                StorageFact::Dense,
            ),
        ]),
    );
    assert!(inferred.diagnostics.is_empty());
    let ValueKindFact::Object(ObjectFact {
        runtime_class: Some(class),
        properties,
        ..
    }) = &inferred.outputs[0].kind
    else {
        panic!("expected typed table output");
    };
    assert!(class.is(runmat_types::standard::TABLE));
    let ValueKindFact::Struct(StructFact { fields, .. }) = &properties["Variables"].kind else {
        panic!("expected statically named table variables");
    };
    assert_eq!(
        fields["Var1"].shape,
        ShapeFact::from(vec![Some(6), Some(1)])
    );
    assert_eq!(
        fields["Var1"].numeric().map(|value| value.class),
        Some(NumericClass::UInt64)
    );
    assert_eq!(fields["Var2"].kind, ValueKindFact::String);
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn preserves_zero_height_and_reports_missing_input() {
    let entry = builtin_catalog_entry_by_name("combinations").expect("combinations entry");
    let inferred = crate::infer_catalog_call(
        entry,
        &request(vec![numeric(
            NumericClass::Int32,
            ShapeFact::from(vec![Some(0), Some(1)]),
        )]),
    );
    let ValueKindFact::Object(table) = &inferred.outputs[0].kind else {
        panic!("expected table");
    };
    let ValueKindFact::Struct(variables) = &table.properties["Variables"].kind else {
        panic!("expected variables");
    };
    assert_eq!(
        variables.fields["Var1"].shape,
        ShapeFact::from(vec![Some(0), Some(1)])
    );

    let missing = crate::infer_catalog_call(entry, &request(Vec::new()));
    assert!(missing
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-COMBINATIONS-ARITY"));
}

#[test]
fn does_not_overstate_runtime_container_boundaries() {
    let entry = builtin_catalog_entry_by_name("combinations").expect("combinations entry");
    let complex = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(1), Some(4)]),
        StorageFact::Dense,
    );
    let sparse = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(3), Some(3)]),
        StorageFact::Sparse,
    );
    let character_matrix = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let inferred =
        crate::infer_catalog_call(entry, &request(vec![complex, sparse, character_matrix]));
    let ValueKindFact::Object(table) = &inferred.outputs[0].kind else {
        panic!("expected table");
    };
    let ValueKindFact::Struct(variables) = &table.properties["Variables"].kind else {
        panic!("expected variables");
    };
    for name in ["Var1", "Var2", "Var3"] {
        assert!(matches!(
            variables.fields[name].kind,
            ValueKindFact::Cell(_)
        ));
        assert_eq!(
            variables.fields[name].shape,
            ShapeFact::from(vec![Some(1), Some(1)])
        );
    }

    let unknown_character_shape = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::Ranked { rank: 2 },
        StorageFact::Dense,
    );
    let inferred = crate::infer_catalog_call(entry, &request(vec![unknown_character_shape]));
    let ValueKindFact::Object(table) = &inferred.outputs[0].kind else {
        panic!("expected table");
    };
    let ValueKindFact::Struct(variables) = &table.properties["Variables"].kind else {
        panic!("expected variables");
    };
    assert!(matches!(
        variables.fields["Var1"].kind,
        ValueKindFact::Unknown
    ));
    assert_eq!(
        variables.fields["Var1"].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
}
