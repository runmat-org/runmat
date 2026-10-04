use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
        arguments,
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn statement_and_bytes_extension_have_distinct_outputs() {
    let args = vec![
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        ),
        ValueFact::scalar(ValueKindFact::String),
    ];
    let statement = inference::infer(&request(args.clone(), 0), &WRITEMATRIX_CATALOG_ENTRY);
    assert!(statement.outputs.is_empty());
    assert!(statement.effects.0.contains(&EffectKind::FilesystemWrite));
    let output = inference::infer(&request(args, 1), &WRITEMATRIX_CATALOG_ENTRY);
    assert_eq!(output.outputs[0].shape, ShapeFact::Scalar);
}

#[test]
fn known_complex_or_ranked_data_is_rejected_without_rejecting_dynamic() {
    let path = ValueFact::scalar(ValueKindFact::String);
    let complex = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    }));
    assert!(inference::infer(
        &request(vec![complex, path.clone()], 0),
        &WRITEMATRIX_CATALOG_ENTRY
    )
    .diagnostics
    .iter()
    .any(|diagnostic| diagnostic.code == "RM-CATALOG-WRITEMATRIX-DATA"));
    let dynamic = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(
        inference::infer(&request(vec![dynamic, path], 0), &WRITEMATRIX_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}
