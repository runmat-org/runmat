use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn file_data_keeps_host_dense_geometry_without_guessing_complexity() {
    let result = inference::infer(
        &request(
            vec![ValueFact::scalar(ValueKindFact::String)],
            vec![LiteralValue::String("values.csv".into())],
        ),
        &CSVREAD_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(result.outputs[0].storage, StorageFact::Dense);
    assert_eq!(result.outputs[0].shape, ShapeFact::from(vec![None, None]));
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
}

#[test]
fn known_bad_filename_and_offset_reject_but_dynamic_contents_remain_possible() {
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let bad = inference::infer(
        &request(
            vec![numeric.clone(), numeric.clone(), numeric.clone()],
            vec![
                LiteralValue::Number(7.0),
                LiteralValue::Number(-1.0),
                LiteralValue::Number(0.0),
            ],
        ),
        &CSVREAD_CATALOG_ENTRY,
    );
    assert!(bad
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-CSVREAD-PATH"));
    assert!(bad
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-CSVREAD-OFFSET"));
    let unknown = inference::infer(
        &request(
            vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            vec![LiteralValue::Unknown],
        ),
        &CSVREAD_CATALOG_ENTRY,
    );
    assert!(unknown.diagnostics.is_empty());
}
