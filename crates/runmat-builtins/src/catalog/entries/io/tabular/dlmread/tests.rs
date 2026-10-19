use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn dlmread_static_result_is_real_host_double_with_unknown_file_dimensions() {
    let call = request(
        vec![ValueFact::scalar(ValueKindFact::String)],
        vec![LiteralValue::String("data.txt".into())],
        1,
    );
    let result = inference::infer(&call, &DLMREAD_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert!(matches!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    ));
    assert_eq!(result.outputs[0].shape, ShapeFact::from(vec![None, None]));
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert!(result.effects.0.contains(&EffectKind::FilesystemRead));
}

#[test]
fn known_invalid_filename_rejects_and_dynamic_control_remains_conservative() {
    let bad = request(
        vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }))],
        vec![LiteralValue::Number(2.0)],
        1,
    );
    assert!(inference::infer(&bad, &DLMREAD_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-DLMREAD-PATH"));
    let dynamic = request(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        vec![
            LiteralValue::String("data.txt".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert!(inference::infer(&dynamic, &DLMREAD_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
