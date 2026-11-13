use super::*;
use runmat_types::{
    CallRequest, CapabilityRequirement, DynamicReason, LiteralContext, LiteralValue,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[test]
fn legacy_discovery_terms_and_related_builtins_remain_available() {
    for keyword in ["read image", "jpg", "bmp", "upload"] {
        assert!(documentation::DOCUMENTATION.keywords.contains(&keyword));
    }
    for related in ["image", "imagesc", "class", "size", "gpuArray"] {
        assert!(documentation::DOCUMENTATION.related.contains(&related));
    }
}

fn request(source: ValueFact, literal: Option<&str>, outputs: RequestedOutputCount) -> CallRequest {
    CallRequest {
        arguments: vec![source],
        literals: LiteralContext::new(vec![literal.map_or(LiteralValue::Unknown, |text| {
            LiteralValue::String(text.into())
        })]),
        outputs: OutputSelection::new(outputs),
    }
}

#[test]
fn external_pixels_keep_unknown_class_but_empty_map_is_exact() {
    let call = request(
        ValueFact::scalar(ValueKindFact::String),
        Some("pixels.ppm"),
        RequestedOutputCount::Exactly(3),
    );
    let result = inference::infer(&call, &IMREAD_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    assert_eq!(result.outputs[0].storage, StorageFact::Dense);
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );
    assert_eq!(result.outputs[2].storage, StorageFact::Dense);
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert!(!result
        .capabilities
        .0
        .contains(&CapabilityRequirement::Network));
    assert_eq!(result.capabilities.0.len(), 1);
}

#[test]
fn url_and_dynamic_source_capabilities_are_not_confused() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let url = inference::infer(
        &request(
            text.clone(),
            Some("HTTPS://example.test/pixels.png"),
            RequestedOutputCount::One,
        ),
        &IMREAD_CATALOG_ENTRY,
    );
    assert!(url.capabilities.0.contains(&CapabilityRequirement::Network));
    assert!(!url
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert_eq!(url.capabilities.0.len(), 1);
    let dynamic = inference::infer(
        &request(text, None, RequestedOutputCount::One),
        &IMREAD_CATALOG_ENTRY,
    );
    assert!(dynamic
        .capabilities
        .0
        .contains(&CapabilityRequirement::Network));
    assert!(dynamic
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert_eq!(dynamic.capabilities.0.len(), 2);
}

#[test]
fn proven_unsupported_scheme_rejects_without_ambient_service_requirement() {
    for (source, code) in [
        ("ftp://example.test/pixels.png", "RM-CATALOG-IMREAD-SCHEME"),
        ("mailto:alice@example.com", "RM-CATALOG-IMREAD-SCHEME"),
        (
            "file://remote.example/pixels.png",
            "RM-CATALOG-IMREAD-FILE-URL",
        ),
    ] {
        let result = inference::infer(
            &request(
                ValueFact::scalar(ValueKindFact::String),
                Some(source),
                RequestedOutputCount::One,
            ),
            &IMREAD_CATALOG_ENTRY,
        );
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == code),
            "{source}"
        );
        assert!(
            !result
                .capabilities
                .0
                .contains(&CapabilityRequirement::Filesystem),
            "{source}"
        );
        assert!(
            !result
                .capabilities
                .0
                .contains(&CapabilityRequirement::Network),
            "{source}"
        );
    }
    assert!(matches!(
        ImreadSourceRoute::parse("C:/pixels.png"),
        ImreadSourceRoute::LocalPath
    ));
    assert!(matches!(
        ImreadSourceRoute::parse("http://"),
        ImreadSourceRoute::LocalPath
    ));
    assert!(matches!(
        ImreadSourceRoute::parse("HTTPS://example.test/pixels.png"),
        ImreadSourceRoute::Http(_)
    ));
    assert!(matches!(
        ImreadSourceRoute::parse("file:///tmp/pixels.png"),
        ImreadSourceRoute::FileUrl(_)
    ));
}

#[test]
fn definitely_invalid_text_shape_and_excess_outputs_diagnose() {
    let source = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(5)]),
        StorageFact::Dense,
    );
    let result = inference::infer(
        &request(source, None, RequestedOutputCount::Exactly(4)),
        &IMREAD_CATALOG_ENTRY,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-IMREAD-TEXT"));
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-IMREAD-OUTPUTS"));
    let unknown = inference::infer(
        &request(
            ValueFact::unknown(DynamicReason::RuntimeValue),
            None,
            RequestedOutputCount::One,
        ),
        &IMREAD_CATALOG_ENTRY,
    );
    assert!(unknown.diagnostics.is_empty());
}

#[test]
fn empty_explicit_format_is_known_invalid_before_io() {
    let call = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::String("missing.png".into()),
            LiteralValue::String(" . ".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer(&call, &IMREAD_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-IMREAD-FORMAT"));
}
