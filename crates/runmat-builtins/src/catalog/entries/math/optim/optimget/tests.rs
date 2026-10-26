use super::*;
use runmat_types::*;
#[test]
fn query_facts_preserve_selected_payload_or_default_without_host_coercion() {
    let selected = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let mut selected = selected;
    selected.residency = ResidencyFact::Device {
        provider: Some("original".into()),
    };
    selected.alias = AliasFact::Unknown;
    let mut fields = std::collections::BTreeMap::new();
    fields.insert("TolX".into(), selected.clone());
    let mut request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true))),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::String("TolX".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer(&request, &OPTIMGET_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0], selected);
    request.literals.literal_args[1] = LiteralValue::String("MaxIter".into());
    let mut fallback = ValueFact::scalar(ValueKindFact::Logical);
    fallback.alias = AliasFact::Unknown;
    request.arguments.push(fallback.clone());
    assert_eq!(
        inference::infer(&request, &OPTIMGET_CATALOG_ENTRY).outputs[0],
        fallback
    );
    request.literals.literal_args[1] = LiteralValue::String("Max".into());
    assert!(!inference::infer(&request, &OPTIMGET_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
