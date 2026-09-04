use std::collections::BTreeMap;

use runmat_types::{
    AliasFact, CallRequest, ClassIdentity, ContiguityFact, LayoutFact, NumericClass, NumericDomain,
    NumericFact, ObjectFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact, ViewFact,
};

use crate::{builtin_catalog_entry_by_name, infer_catalog_call};

fn infer(name: &str, input: ValueFact) -> runmat_types::CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name(name).expect("exponential catalog entry"),
        &CallRequest {
            arguments: vec![input],
            literals: Default::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

#[test]
fn floating_facts_are_preserved_and_integer_conversion_loses_device_certainty() {
    let shape = ShapeFact::from(vec![Some(3), Some(2)]);
    let mut complex_single = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        shape.clone(),
        StorageFact::Dense,
    );
    complex_single.residency = ResidencyFact::Device {
        provider: Some("wgpu".into()),
    };
    let preserve = infer("exp", complex_single);
    assert!(preserve.diagnostics.is_empty());
    assert_eq!(
        preserve.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );
    assert_eq!(preserve.outputs[0].shape, shape);
    assert_eq!(
        preserve.outputs[0].residency,
        ResidencyFact::Device {
            provider: Some("wgpu".into())
        }
    );
    assert_eq!(preserve.outputs[0].layout, LayoutFact::ColumnMajor);
    assert_eq!(preserve.outputs[0].contiguity, ContiguityFact::Contiguous);
    assert_eq!(preserve.outputs[0].view, ViewFact::Materialized);
    assert_eq!(preserve.outputs[0].alias, AliasFact::Unique);

    let mut integer = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt16,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    );
    integer.residency = ResidencyFact::Device {
        provider: Some("integer-provider".into()),
    };
    let converted = infer("exp", integer);
    assert_eq!(
        converted.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(converted.outputs[0].residency, ResidencyFact::Unknown);
}

#[test]
fn sparse_zero_semantics_are_operation_specific() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(5), Some(3)]),
        StorageFact::Sparse,
    );
    let exp = infer("exp", input.clone());
    let expm1 = infer("expm1", input);
    assert!(exp.diagnostics.is_empty());
    assert!(expm1.diagnostics.is_empty());
    assert_eq!(exp.outputs[0].kind, expm1.outputs[0].kind);
    assert_eq!(exp.outputs[0].shape, expm1.outputs[0].shape);
    assert_eq!(exp.outputs[0].storage, StorageFact::Dense);
    assert_eq!(expm1.outputs[0].storage, StorageFact::Sparse);
    assert_eq!(expm1.outputs[0].view, ViewFact::Materialized);
    assert_eq!(expm1.outputs[0].alias, AliasFact::Unique);
}

#[test]
fn integer_complex_tabular_and_overloaded_inputs_follow_distinct_contracts() {
    let integer = infer(
        "expm1",
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), Some(4)]),
            StorageFact::Dense,
        ),
    );
    assert_eq!(
        integer.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );

    let complex_integer = infer(
        "expm1",
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int32,
            domain: NumericDomain::Complex,
        })),
    );
    assert!(complex_integer
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-EXPONENTIAL-COMPLEX-INTEGER"));

    let table = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::TABLE.owned()),
            properties: BTreeMap::from([(
                "Variables".into(),
                ValueFact::scalar(ValueKindFact::Logical),
            )]),
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let table_output = infer("expm1", table);
    let ValueKindFact::Object(table_output) = &table_output.outputs[0].kind else {
        panic!("expected tabular object fact");
    };
    assert_eq!(
        table_output.runtime_class,
        Some(runmat_types::standard::TABLE.owned())
    );
    assert!(table_output.properties.is_empty());
    assert!(!table_output.properties_complete);

    let user_object = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(ClassIdentity::new("UserClass").unwrap()),
        properties: BTreeMap::new(),
        properties_complete: false,
        handle_semantics: None,
    }));
    let overloaded = infer("expm1", user_object);
    assert_eq!(overloaded.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(overloaded.outputs[0].shape, ShapeFact::Scalar);
}
