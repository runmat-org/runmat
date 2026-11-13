use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}
fn numeric(class: NumericClass, domain: NumericDomain, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        shape,
        StorageFact::Dense,
    )
}
#[test]
fn sparse_facts_follow_storage_class_shape_and_actual_forms() {
    for input in [
        ValueFact::unknown(DynamicReason::RuntimeValue),
        ValueFact::scalar(ValueKindFact::Logical),
    ] {
        let result = inference::infer(
            &request(vec![input.clone()], vec![LiteralValue::Unknown]),
            &SPARSE_CATALOG_ENTRY,
        );
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, input.kind);
        assert_eq!(result.outputs[0].storage, StorageFact::Sparse);
        assert_eq!(result.outputs[0].shape.rank(), Some(2));
    }
    for (class, domain) in [
        (NumericClass::Single, NumericDomain::Real),
        (NumericClass::Double, NumericDomain::Complex),
        (NumericClass::UInt64, NumericDomain::Real),
    ] {
        let input = numeric(class, domain, ShapeFact::from(vec![Some(2), Some(3)]));
        let result = inference::infer(
            &request(vec![input.clone()], vec![LiteralValue::Unknown]),
            &SPARSE_CATALOG_ENTRY,
        );
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, input.kind);
        assert_eq!(result.outputs[0].shape, input.shape);
        assert_eq!(result.outputs[0].storage, StorageFact::Sparse);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    }
    let size = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let unknown = request(
        vec![size.clone(), size.clone()],
        vec![LiteralValue::Unknown; 2],
    );
    let result = inference::infer(&unknown, &SPARSE_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].shape, ShapeFact::from(vec![None, None]));
    let logical = request(
        vec![
            size.clone(),
            size.clone(),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![
            LiteralValue::Number(2.0),
            LiteralValue::Number(3.0),
            LiteralValue::String("logical".into()),
        ],
    );
    let result = inference::infer(&logical, &SPARSE_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    for bad in [
        request(vec![], vec![]),
        request(vec![size.clone(); 4], vec![LiteralValue::Unknown; 4]),
        request(
            vec![size.clone(), size],
            vec![LiteralValue::Number(-1.0), LiteralValue::Number(3.0)],
        ),
        request(
            vec![numeric(
                NumericClass::Int32,
                NumericDomain::Complex,
                ShapeFact::Scalar,
            )],
            vec![LiteralValue::Unknown],
        ),
    ] {
        assert!(!inference::infer(&bad, &SPARSE_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }
}
