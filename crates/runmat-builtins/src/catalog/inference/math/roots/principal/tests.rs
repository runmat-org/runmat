use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn infer(input: ValueFact) -> runmat_types::CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name("sqrt").expect("catalog entry"),
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

#[test]
fn preserves_floating_classes_and_promotes_extension_inputs() {
    for (kind, expected) in [
        (
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Complex,
            }),
            NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Complex,
            },
        ),
        (
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt16,
                domain: NumericDomain::Real,
            }),
            NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            },
        ),
        (
            ValueKindFact::Logical,
            NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            },
        ),
    ] {
        let output = infer(ValueFact::proven(
            kind,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        ));
        assert!(output.diagnostics.is_empty());
        assert_eq!(output.outputs[0].kind, ValueKindFact::Numeric(expected));
        assert_eq!(
            output.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
    }
}

#[test]
fn rejects_sparse_and_complex_integer_inputs() {
    let sparse = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Sparse,
    );
    assert!(!infer(sparse).diagnostics.is_empty());
    let complex_integer = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Int32,
        domain: NumericDomain::Complex,
    }));
    assert!(!infer(complex_integer).diagnostics.is_empty());
}
