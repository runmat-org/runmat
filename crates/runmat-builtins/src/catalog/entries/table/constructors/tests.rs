use super::*;
use crate::*;
use runmat_types::*;
fn call(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}
#[test]
fn constructors_preserve_distinct_object_and_code_geometry() {
    for (entry, class) in [
        (&CATEGORICAL_CATALOG_ENTRY, standard::CATEGORICAL),
        (&ORDINAL_CATALOG_ENTRY, standard::CATEGORICAL),
        (&DICTIONARY_CATALOG_ENTRY, standard::DICTIONARY),
        (&TABLE_CATALOG_ENTRY, standard::TABLE),
    ] {
        let result = infer_catalog_call(
            entry,
            &call(if entry.identity.name == "ordinal" {
                vec![ValueFact::proven(
                    ValueKindFact::String,
                    ShapeFact::from(vec![Some(2), Some(1), Some(3)]),
                    StorageFact::Dense,
                )]
            } else {
                vec![]
            }),
        );
        let ValueKindFact::Object(object) = &result.outputs[0].kind else {
            panic!("object contract")
        };
        assert_eq!(object.runtime_class, Some(class.owned()));
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        assert!(result.effects.0.contains(&EffectKind::MaySuspend));
        assert_eq!(entry.placement.fusion, BuiltinFusionPolicy::Boundary);
    }
    let result = infer_catalog_call(&CATEGORICAL_CATALOG_ENTRY, &call(vec![]));
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("categorical")
    };
    assert_eq!(
        object.properties["Codes"].shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );
    let result = infer_catalog_call(&ORDINAL_CATALOG_ENTRY, &call(vec![]));
    assert!(!result.diagnostics.is_empty());
}
#[test]
fn table_schema_retains_native_columns_and_rejects_mismatched_rows() {
    let column = |rows, class| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(rows), Some(1)]),
            StorageFact::Dense,
        )
    };
    let result = infer_catalog_call(
        &TABLE_CATALOG_ENTRY,
        &call(vec![
            column(3, NumericClass::UInt64),
            column(3, NumericClass::Single),
        ]),
    );
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("table")
    };
    let ValueKindFact::Struct(columns) = &object.properties["__table_variables"].kind else {
        panic!("schema")
    };
    assert_eq!(
        columns.fields["Var1"].kind,
        column(3, NumericClass::UInt64).kind
    );
    assert_eq!(
        columns.fields["Var2"].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert!(!infer_catalog_call(
        &TABLE_CATALOG_ENTRY,
        &call(vec![
            column(2, NumericClass::UInt64),
            column(3, NumericClass::Single)
        ])
    )
    .diagnostics
    .is_empty());
}

#[test]
fn dictionary_scalar_entry_facts_preserve_single_class() {
    let value = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let result = infer_catalog_call(&DICTIONARY_CATALOG_ENTRY, &call(vec![value.clone(), value]));
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("dictionary")
    };
    for field in ["Keys", "Values"] {
        let ValueKindFact::Cell(cell) = &object.properties[field].kind else {
            panic!("cell entries")
        };
        assert_eq!(
            cell.element.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real
            })
        );
        assert_eq!(cell.element.shape, ShapeFact::Scalar);
    }
}
