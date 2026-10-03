use super::*;
use runmat_types::{
    CallRequest, CapabilityRequirement, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn image(class: NumericClass, shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}

fn call(arguments: Vec<ValueFact>, outputs: RequestedOutputCount) -> CallRequest {
    CallRequest {
        literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
        arguments,
        outputs: OutputSelection::new(outputs),
    }
}

#[test]
fn direct_image_has_no_output_and_requires_destination_filesystem() {
    let request = call(
        vec![
            image(NumericClass::UInt8, vec![Some(2), Some(3), Some(3)]),
            ValueFact::scalar(ValueKindFact::String),
        ],
        RequestedOutputCount::Zero,
    );
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert!(result.outputs.is_empty());
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
}

#[test]
fn invalid_class_map_and_output_are_diagnosed_without_inventing_values() {
    let request = call(
        vec![
            image(NumericClass::Int32, vec![Some(2), Some(3)]),
            image(NumericClass::UInt8, vec![Some(2), Some(3)]),
            ValueFact::scalar(ValueKindFact::String),
        ],
        RequestedOutputCount::One,
    );
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-IMAGE"));
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-MAP"));
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-OUTPUTS"));
}

#[test]
fn known_unsupported_channels_and_empty_destination_diagnose() {
    let mut request = call(
        vec![
            image(NumericClass::Double, vec![Some(2), Some(2), Some(5)]),
            ValueFact::scalar(ValueKindFact::String),
        ],
        RequestedOutputCount::Zero,
    );
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::String("  ".into()),
    ]);
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-IMAGE"));
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-FILENAME"));
}

#[test]
fn option_aliases_share_the_runtime_decoder() {
    assert_eq!(
        ImwriteOption::from_name("Write_Mode"),
        Some(ImwriteOption::WriteMode)
    );
    assert_eq!(
        ImwriteOption::from_name("Delay-Time"),
        Some(ImwriteOption::DelayTime)
    );
    let mut request = call(
        vec![
            image(NumericClass::UInt8, vec![Some(2), Some(2)]),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        RequestedOutputCount::Zero,
    );
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::String("x.png".into()),
        LiteralValue::String("Compression".into()),
    ]);
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-OPTION"));
}
