use super::*;
use runmat_types::{
    CallRequest, CapabilityRequirement, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

#[test]
fn legacy_writer_discovery_terms_remain_available() {
    for keyword in ["write image", "jpg", "indexed image"] {
        assert!(documentation::DOCUMENTATION.keywords.contains(&keyword));
    }
}

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
    assert_eq!(ImwriteFormat::from_hint(".JPEG"), Some(ImwriteFormat::Jpeg));
    assert_eq!(
        ImwriteWriteMode::from_name(" append "),
        Some(ImwriteWriteMode::Append)
    );
    assert!(ImwriteOption::LoopCount.validate_numeric(0.5).is_err());
    assert!(ImwriteOption::DelayTime.validate_numeric(1e300).is_err());
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

#[test]
fn every_proven_option_pair_and_format_channel_rule_is_checked() {
    let text = || ValueFact::scalar(ValueKindFact::String);
    let mut request = call(
        vec![
            image(NumericClass::Double, vec![Some(2), Some(2), Some(3)]),
            text(),
            text(),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            text(),
            text(),
        ],
        RequestedOutputCount::Zero,
    );
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::String("x.gif".into()),
        LiteralValue::String("DelayTime".into()),
        LiteralValue::Unknown,
        LiteralValue::String("Compression".into()),
        LiteralValue::Unknown,
    ]);
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-OPTION" && d.argument == Some(4)));
    request.arguments.truncate(4);
    request.literals.literal_args.truncate(4);
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-IMAGE"));

    request.arguments.push(text());
    request.arguments.push(text());
    request
        .literals
        .literal_args
        .push(LiteralValue::String("Compression".into()));
    request.literals.literal_args.push(LiteralValue::Unknown);

    request.arguments.pop();
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-OPTION" && d.argument == Some(4)));

    request.arguments = vec![
        image(NumericClass::UInt16, vec![Some(2), Some(2)]),
        text(),
        text(),
    ];
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::String("x.dat".into()),
        LiteralValue::String("jpg".into()),
    ]);
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-IMAGE"));

    request.arguments = vec![
        image(NumericClass::UInt8, vec![Some(1), Some(1)]),
        text(),
        text(),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    ];
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::String("x.jpg".into()),
        LiteralValue::String("Quality".into()),
        LiteralValue::Number(101.0),
    ]);
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-OPTION" && d.argument == Some(3)));

    request.arguments[3] = text();
    request.literals.literal_args[3] = LiteralValue::String("bad".into());
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-OPTION" && d.argument == Some(3)));

    request.literals.literal_args[2] = LiteralValue::String("WriteMode".into());
    request.arguments[3] = image(NumericClass::Double, vec![Some(1), Some(1)]);
    request.literals.literal_args[3] = LiteralValue::Number(1.0);
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-OPTION" && d.argument == Some(3)));

    // The second value might instead be an indexed colormap. No guessed
    // filename or option positions are allowed from that unresolved shape.
    request.arguments[1] = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(!result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-OPTION"));
}

#[test]
fn complex_pixels_colormap_alpha_and_mismatched_alpha_shape_diagnose() {
    let complex = |shape| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Complex,
            }),
            ShapeFact::from(shape),
            StorageFact::Dense,
        )
    };
    let text = || ValueFact::scalar(ValueKindFact::String);
    let result = inference::infer(
        &call(
            vec![complex(vec![Some(2), Some(2)]), text()],
            RequestedOutputCount::Zero,
        ),
        &IMWRITE_CATALOG_ENTRY,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-IMAGE"));
    let result = inference::infer(
        &call(
            vec![
                image(NumericClass::UInt8, vec![Some(2), Some(2)]),
                complex(vec![Some(2), Some(3)]),
                text(),
            ],
            RequestedOutputCount::Zero,
        ),
        &IMWRITE_CATALOG_ENTRY,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-MAP"));

    let mut request = call(
        vec![
            image(NumericClass::UInt8, vec![Some(2), Some(2)]),
            text(),
            text(),
            complex(vec![Some(2), Some(2)]),
        ],
        RequestedOutputCount::Zero,
    );
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::String("x.png".into()),
        LiteralValue::String("Alpha".into()),
        LiteralValue::Unknown,
    ]);
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-OPTION" && d.argument == Some(3)));
    request.arguments[3] = image(NumericClass::Double, vec![Some(1), Some(2)]);
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-OPTION" && d.argument == Some(3)));
    request.arguments[3] = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(!result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-IMWRITE-OPTION" && d.argument == Some(3)));
}

#[test]
fn logical_alpha_matches_runtime_class_and_still_requires_image_geometry() {
    let logical = |shape| {
        ValueFact::proven(
            ValueKindFact::Logical,
            ShapeFact::from(shape),
            StorageFact::Dense,
        )
    };
    let mut request = call(
        vec![
            image(NumericClass::UInt8, vec![Some(2), Some(3)]),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            logical(vec![Some(2), Some(3)]),
        ],
        RequestedOutputCount::Zero,
    );
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::String("x.png".into()),
        LiteralValue::String("Alpha".into()),
        LiteralValue::Unknown,
    ]);
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);

    request.arguments[3] = logical(vec![Some(1), Some(3)]);
    let result = inference::infer(&request, &IMWRITE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-IMWRITE-OPTION"
            && diagnostic.argument == Some(3)));
}
