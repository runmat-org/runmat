use crate::{infer_catalog_call, BuiltinCatalogEntry};
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
pub(super) fn check(entry: &BuiltinCatalogEntry, indices: bool) {
    for (text, expected) in [
        (
            "\u{2003} REVERSE\t",
            Some(super::CumulativeControl::Direction(
                super::CumulativeScanDirection::Reverse,
            )),
        ),
        (
            " omitmissing ",
            Some(super::CumulativeControl::Missing(
                super::CumulativeMissingMode::Omit,
            )),
        ),
        ("forward reverse", None),
    ] {
        assert_eq!(super::CumulativeControl::from_public_option(text), expected);
        let chars: Vec<char> = text.chars().collect();
        assert_eq!(
            super::CumulativeControl::from_public_characters(chars.iter().copied()),
            expected
        );
        assert_eq!(
            super::CumulativeControl::from_public_code_units(
                text.encode_utf16().collect::<Vec<_>>().into_iter()
            ),
            expected
        );
    }
    assert_eq!(
        super::CumulativeControl::from_public_code_units([0xd800].into_iter()),
        None
    );
    for class in [
        NumericClass::Single,
        NumericClass::Double,
        NumericClass::UInt64,
    ] {
        for domain in [NumericDomain::Real, NumericDomain::Complex] {
            if class == NumericClass::UInt64 && domain == NumericDomain::Complex {
                continue;
            }
            let value = ValueFact::proven(
                ValueKindFact::Numeric(NumericFact { class, domain }),
                ShapeFact::from(vec![Some(3), Some(1)]),
                StorageFact::Dense,
            );
            let request = CallRequest {
                arguments: vec![value.clone()],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            };
            let inferred = infer_catalog_call(entry, &request);
            assert!(inferred.diagnostics.is_empty());
            assert_eq!(inferred.outputs[0].kind, value.kind);
            assert_eq!(inferred.outputs[0].shape, value.shape);
        }
    }
    let value = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(0), Some(3)]),
        StorageFact::Dense,
    );
    let mut request = CallRequest {
        arguments: vec![
            value.clone(),
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
        ],
        literals: LiteralContext {
            literal_args: vec![
                LiteralValue::Unknown,
                LiteralValue::String("reverse".into()),
                LiteralValue::String("omitmissing".into()),
            ],
        },
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inferred = infer_catalog_call(entry, &request);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs[0].shape, value.shape);
    if indices {
        request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
        let pair = infer_catalog_call(entry, &request);
        assert!(pair.diagnostics.is_empty());
        assert_eq!(pair.outputs.len(), 2);
        assert_eq!(pair.outputs[1].shape, value.shape);
        assert_eq!(
            pair.outputs[1].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        );
    }
    request.outputs =
        OutputSelection::new(RequestedOutputCount::Exactly(if indices { 3 } else { 2 }));
    assert!(!infer_catalog_call(entry, &request).diagnostics.is_empty());
    request.outputs = OutputSelection::new(RequestedOutputCount::One);
    request.literals.literal_args[2] = LiteralValue::String("forward".into());
    assert!(!infer_catalog_call(entry, &request).diagnostics.is_empty());
    request.arguments.clear();
    assert!(!infer_catalog_call(entry, &request).diagnostics.is_empty());
    assert_eq!(
        entry.descriptor.signatures[0].outputs.len(),
        if indices { 2 } else { 1 }
    );
    assert_eq!(entry.bindings.len(), 1);
}
