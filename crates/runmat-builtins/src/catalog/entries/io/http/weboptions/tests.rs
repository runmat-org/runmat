use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn typed_host_options_output_and_literal_name_admission() {
    let good = request(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        ],
        vec![
            LiteralValue::String("tImEoUt".into()),
            LiteralValue::Number(10.0),
        ],
    );
    let result = inference::infer(&good, &WEBOPTIONS_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    let ValueKindFact::Struct(structure) = &result.outputs[0].kind else {
        panic!("expected options struct")
    };
    assert!(structure.fields_complete);
    assert_eq!(structure.fields.len(), 11);
    assert_eq!(
        structure.fields["Timeout"].numeric().unwrap().class,
        NumericClass::Double
    );
    assert!(result.capabilities.0.is_empty());

    let unknown = request(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        vec![
            LiteralValue::String("CharacterEncoding".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(!inference::infer(&unknown, &WEBOPTIONS_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn incomplete_pair_and_invalid_name_are_definite_but_dynamic_name_is_not() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let pair = request(
        vec![text.clone()],
        vec![LiteralValue::String("Timeout".into())],
    );
    assert!(!inference::infer(&pair, &WEBOPTIONS_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let numeric = request(
        vec![
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            text.clone(),
        ],
        vec![LiteralValue::Number(7.0), LiteralValue::String("x".into())],
    );
    assert!(!inference::infer(&numeric, &WEBOPTIONS_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let dynamic = request(
        vec![ValueFact::unknown(DynamicReason::RuntimeValue), text],
        vec![LiteralValue::Unknown, LiteralValue::Unknown],
    );
    assert!(inference::infer(&dynamic, &WEBOPTIONS_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn known_option_value_class_is_checked_without_rejecting_dynamic_values() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let logical = ValueFact::scalar(ValueKindFact::Logical);
    let bad_timeout = request(
        vec![text.clone(), logical.clone()],
        vec![
            LiteralValue::String("Timeout".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(!inference::infer(&bad_timeout, &WEBOPTIONS_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let good_key = request(
        vec![text.clone(), logical],
        vec![
            LiteralValue::String("KeyValue".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(inference::infer(&good_key, &WEBOPTIONS_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let unknown_timeout = request(
        vec![text, ValueFact::unknown(DynamicReason::RuntimeValue)],
        vec![
            LiteralValue::String("Timeout".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(
        inference::infer(&unknown_timeout, &WEBOPTIONS_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}
