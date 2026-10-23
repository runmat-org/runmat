use super::*;
use crate::infer_catalog_call;
use runmat_types::*;

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}
fn call(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::default(),
    }
}
fn text(value: &str) -> LiteralValue {
    LiteralValue::String(value.into())
}

#[test]
fn descriptor_retains_exact_payloads_and_only_constructs_metadata() {
    let integer = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let result = infer_catalog_call(
        &FILE_DATASTORE_ENTRY,
        &call(
            vec![
                integer.clone(),
                ValueFact::scalar(ValueKindFact::String),
                integer.clone(),
            ],
            vec![
                LiteralValue::Unknown,
                text("ReadFcn"),
                LiteralValue::Unknown,
            ],
        ),
    );
    assert!(result.diagnostics.is_empty());
    let output = &result.outputs[0];
    assert_eq!(output.shape, ShapeFact::Scalar);
    assert_eq!(output.residency, ResidencyFact::Host);
    let ValueKindFact::Object(object) = &output.kind else {
        panic!("object")
    };
    assert_eq!(
        object.runtime_class,
        Some(StaticClassIdentity::new("fileDatastore").owned())
    );
    assert!(object.properties_complete);
    assert_eq!(object.properties.len(), 5);
    assert_eq!(object.handle_semantics, Some(false));
    for field in ["Files", "ReadFcn"] {
        assert_eq!(object.properties[field].kind, integer.kind);
        assert_eq!(object.properties[field].shape, integer.shape);
    }
    assert_eq!(
        object.properties["IncludeSubfolders"].kind,
        ValueKindFact::Logical
    );
    assert_eq!(
        FILE_DATASTORE_ENTRY.contract.maturity,
        crate::BuiltinContractMaturity::Incomplete
    );
    assert!(FILE_DATASTORE_ENTRY.contract.capabilities.is_empty());
    assert_eq!(
        FILE_DATASTORE_ENTRY.contract.effects,
        &[EffectKind::MaySuspend, EffectKind::MayThrow]
    );
}

#[test]
fn typed_option_controls_and_fixed_outputs_agree() {
    assert_eq!(
        FileDatastoreOption::parse("rEaDfCn"),
        Some(FileDatastoreOption::ReadFcn)
    );
    assert_eq!(
        FileDatastoreReadMode::parse("PARTIALFILE"),
        Some(FileDatastoreReadMode::PartialFile)
    );
    assert_eq!(FileDatastoreReadMode::parse("bytes"), None);
    for (name, value) in [
        ("ReadMode", "bytes"),
        ("IncludeSubfolders", "maybe"),
        ("PreviewFcn", "anything"),
    ] {
        let request = call(
            vec![
                unknown(),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            vec![LiteralValue::Unknown, text(name), text(value)],
        );
        assert!(
            !infer_catalog_call(&FILE_DATASTORE_ENTRY, &request)
                .diagnostics
                .is_empty(),
            "{name}"
        );
    }
    for count in [0, 2] {
        assert!(!infer_catalog_call(
            &FILE_DATASTORE_ENTRY,
            &call(vec![unknown(); count], Vec::new())
        )
        .diagnostics
        .is_empty());
    }
    let mut request = call(vec![unknown()], Vec::new());
    request.outputs = OutputSelection::new(RequestedOutputCount::Zero);
    let result = infer_catalog_call(&FILE_DATASTORE_ENTRY, &request);
    assert!(result.outputs.is_empty());
    assert!(result.diagnostics.is_empty());
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(!infer_catalog_call(&FILE_DATASTORE_ENTRY, &request)
        .diagnostics
        .is_empty());
}
