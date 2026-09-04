use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn infer(input: ValueFact) -> runmat_types::CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name("realsqrt").expect("catalog entry"),
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

#[test]
fn preserves_floating_class_shape_and_sparse_storage() {
    for (class, storage) in [
        (NumericClass::Double, StorageFact::Dense),
        (NumericClass::Single, StorageFact::Sparse),
    ] {
        let output = infer(ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(3), Some(2)]),
            storage,
        ));
        assert!(output.diagnostics.is_empty());
        assert_eq!(
            output.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            })
        );
        assert_eq!(output.outputs[0].storage, storage);
    }
}

#[test]
fn rejects_integer_complex_and_logical_inputs() {
    for kind in [
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int64,
            domain: NumericDomain::Real,
        }),
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ValueKindFact::Logical,
    ] {
        assert!(!infer(ValueFact::scalar(kind)).diagnostics.is_empty());
    }
}
