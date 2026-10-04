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
fn known_output_type_and_unknown_control_do_not_invent_the_same_class() {
    let path = ValueFact::scalar(ValueKindFact::String);
    let name = ValueFact::scalar(ValueKindFact::String);
    let value = ValueFact::scalar(ValueKindFact::String);
    let known = inference::infer(
        &request(
            vec![path.clone(), name.clone(), value.clone()],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("OutputType".into()),
                LiteralValue::String("uint64".into()),
            ],
            1,
        ),
        &READMATRIX_CATALOG_ENTRY,
    );
    assert_eq!(
        known.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        })
    );
    let unknown = inference::infer(
        &request(
            vec![path, name, value],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("OutputType".into()),
                LiteralValue::Unknown,
            ],
            1,
        ),
        &READMATRIX_CATALOG_ENTRY,
    );
    assert_eq!(unknown.outputs[0].kind, ValueKindFact::Unknown);

    let options = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        std::collections::BTreeMap::new(),
        false,
    )));
    let structured = inference::infer(
        &request(
            vec![ValueFact::scalar(ValueKindFact::String), options],
            vec![LiteralValue::Unknown, LiteralValue::Unknown],
            1,
        ),
        &READMATRIX_CATALOG_ENTRY,
    );
    assert_eq!(structured.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn malformed_pair_and_known_bad_class_have_catalog_diagnostics() {
    let path = ValueFact::scalar(ValueKindFact::String);
    let option = ValueFact::scalar(ValueKindFact::String);
    let dangling = inference::infer(
        &request(
            vec![path.clone(), option.clone()],
            vec![LiteralValue::Unknown, LiteralValue::Unknown],
            1,
        ),
        &READMATRIX_CATALOG_ENTRY,
    );
    assert!(dangling
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-READMATRIX-ARITY"));
    let invalid = inference::infer(
        &request(
            vec![path, option.clone(), option],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("OutputType".into()),
                LiteralValue::String("quaternion".into()),
            ],
            1,
        ),
        &READMATRIX_CATALOG_ENTRY,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-READMATRIX-OUTPUT-TYPE"));
}
