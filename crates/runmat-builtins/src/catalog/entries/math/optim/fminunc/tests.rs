use super::*;
use runmat_types::*;
#[test]
fn selected_algorithm_does_not_forge_numeric_hinfo_or_nonempty_model_facts() {
    let x = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let mut request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::UnresolvedCallable), x],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(6)),
    };
    let result = inference::infer(&request, &FMINUNC_CATALOG_ENTRY);
    assert!(matches!(result.outputs[5].kind, ValueKindFact::Numeric(_)));
    assert_eq!(result.outputs[5].shape.element_count(), Some(4));
    let mut fields = std::collections::BTreeMap::new();
    fields.insert("Algorithm".into(), ValueFact::scalar(ValueKindFact::String));
    request
        .arguments
        .push(ValueFact::scalar(ValueKindFact::Struct(
            StructFact::scalar(fields, true),
        )));
    let result = inference::infer(&request, &FMINUNC_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[5].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[5].shape, ShapeFact::Unknown);

    for field in [
        crate::OptimOptionField::HessianApproximation,
        crate::OptimOptionField::HessUpdate,
    ] {
        let mut fields = std::collections::BTreeMap::new();
        fields.insert(
            field.name().into(),
            ValueFact::scalar(ValueKindFact::String),
        );
        request.arguments[2] =
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
        let result = inference::infer(&request, &FMINUNC_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert!(matches!(result.outputs[5].kind, ValueKindFact::Numeric(_)));
        assert_eq!(result.outputs[5].shape, ShapeFact::Unknown);
        assert_eq!(result.outputs[5].storage, StorageFact::Dense);
    }
    request.arguments.pop();
    request.arguments[1] = super::super::nonlinear_facts::scalar();
    let result = inference::infer(&request, &FMINUNC_CATALOG_ENTRY);
    assert_eq!(result.outputs[5].shape.element_count(), Some(1));
    assert_eq!(result.outputs[5].storage, StorageFact::Dense);
}
