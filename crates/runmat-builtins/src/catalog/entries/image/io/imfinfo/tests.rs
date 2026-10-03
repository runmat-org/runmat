use super::*;
use runmat_types::{
    CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn request(
    path: ValueFact,
    format: Option<ValueFact>,
    outputs: RequestedOutputCount,
) -> CallRequest {
    let mut arguments = vec![path];
    arguments.extend(format);
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(outputs),
    }
}

#[test]
fn metadata_fields_are_typed_and_text_shape_is_checked() {
    let inferred = inference::infer(
        &request(
            ValueFact::scalar(ValueKindFact::String),
            None,
            RequestedOutputCount::One,
        ),
        &IMFINFO_CATALOG_ENTRY,
    );
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    let ValueKindFact::Struct(fields) = &inferred.outputs[0].kind else {
        panic!("expected struct")
    };
    assert!(fields.fields_complete);
    assert_eq!(fields.fields["Width"].shape, ShapeFact::Scalar);
    assert!(fields.fields.contains_key("Format"));
    let invalid = inference::infer(
        &request(
            ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(2), Some(3)]),
                StorageFact::Dense,
            ),
            None,
            RequestedOutputCount::One,
        ),
        &IMFINFO_CATALOG_ENTRY,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-IMFINFO-TEXT"));
    let unknown = inference::infer(
        &request(
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
            None,
            RequestedOutputCount::One,
        ),
        &IMFINFO_CATALOG_ENTRY,
    );
    assert!(unknown.diagnostics.is_empty());
}

#[test]
fn output_count_and_format_argument_have_separate_facts() {
    let inferred = inference::infer(
        &request(
            ValueFact::scalar(ValueKindFact::String),
            Some(ValueFact::scalar(ValueKindFact::String)),
            RequestedOutputCount::Exactly(2),
        ),
        &IMFINFO_CATALOG_ENTRY,
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-IMFINFO-OUTPUTS"));
}
