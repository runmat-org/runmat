use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn dlmwrite_statement_and_extension_output_shapes_are_distinct() {
    let args = vec![
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        ),
    ];
    let statement = inference::infer(&request(args.clone(), 0), &DLMWRITE_CATALOG_ENTRY);
    assert!(statement.outputs.is_empty());
    assert!(statement.effects.0.contains(&EffectKind::FilesystemWrite));
    let bytes = inference::infer(&request(args, 1), &DLMWRITE_CATALOG_ENTRY);
    assert_eq!(bytes.outputs[0].shape, ShapeFact::Scalar);
    assert!(matches!(bytes.outputs[0].kind, ValueKindFact::Numeric(_)));
}

#[test]
fn definitely_complex_data_is_rejected_but_dynamic_remains_possible() {
    let path = ValueFact::scalar(ValueKindFact::String);
    let complex = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    }));
    assert!(inference::infer(
        &request(vec![path.clone(), complex], 0),
        &DLMWRITE_CATALOG_ENTRY
    )
    .diagnostics
    .iter()
    .any(|d| d.code == "RM-CATALOG-DLMWRITE-DATA"));
    let dynamic = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(
        inference::infer(&request(vec![path, dynamic], 0), &DLMWRITE_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}
