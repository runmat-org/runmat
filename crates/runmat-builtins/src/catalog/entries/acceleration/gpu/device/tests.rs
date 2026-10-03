use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn device_is_a_host_struct_with_only_proven_provider_fields() {
    let result = inference::infer(&request(vec![]), &GPU_DEVICE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    let ValueKindFact::Struct(fields) = &result.outputs[0].kind else {
        panic!("gpuDevice should infer a struct")
    };
    assert!(!fields.fields_complete);
    let known = &fields.fields;
    for name in [
        "device_id",
        "index",
        "name",
        "vendor",
        "precision",
        "supports_double",
    ] {
        assert!(known.contains_key(name), "missing {name}");
    }
    assert!(!known.contains_key("backend"));
    assert!(!known.contains_key("memory_bytes"));
    assert!(
        matches!(&known["device_id"].kind, ValueKindFact::Numeric(fact) if fact.class == NumericClass::UInt32)
    );
    assert!(
        matches!(&known["index"].kind, ValueKindFact::Numeric(fact) if fact.class == NumericClass::UInt32)
    );
}

#[test]
fn device_static_arity_and_nonscalar_selector_are_rejected() {
    let scalar = ValueFact::scalar(ValueKindFact::Unknown);
    let many = inference::infer(
        &request(vec![scalar.clone(), scalar]),
        &GPU_DEVICE_CATALOG_ENTRY,
    );
    assert!(many
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.GPU_DEVICE.TOO_MANY_INPUTS"));
    let matrix = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Dense,
    );
    let nonscalar = inference::infer(&request(vec![matrix]), &GPU_DEVICE_CATALOG_ENTRY);
    assert!(nonscalar
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.GPU_DEVICE.INVALID_INDEX"));
}

#[test]
fn logical_first_selector_still_has_semantic_arity_diagnostic() {
    let logical = ValueFact::scalar(ValueKindFact::Logical);
    let extra = ValueFact::scalar(ValueKindFact::Unknown);
    let result = inference::infer(&request(vec![logical, extra]), &GPU_DEVICE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == GPU_DEVICE_ERROR_TOO_MANY_INPUTS.code));
}

#[test]
fn selector_known_empty_invalid_and_unavailable_have_stable_diagnostics() {
    let numeric = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    });
    let empty = ValueFact::proven(
        numeric.clone(),
        ShapeFact::from(vec![Some(0), Some(0)]),
        StorageFact::Dense,
    );
    let result = inference::infer(&request(vec![empty]), &GPU_DEVICE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.GPU_DEVICE.RESET_NOT_SUPPORTED"));
    let mut invalid = request(vec![ValueFact::scalar(numeric.clone())]);
    invalid.literals = LiteralContext::new(vec![LiteralValue::Number(0.0)]);
    let result = inference::infer(&invalid, &GPU_DEVICE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.GPU_DEVICE.INVALID_INDEX"));
    let mut unavailable = request(vec![ValueFact::scalar(numeric)]);
    unavailable.literals = LiteralContext::new(vec![LiteralValue::Number(2.0)]);
    let result = inference::infer(&unavailable, &GPU_DEVICE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.GPU_DEVICE.DEVICE_NOT_AVAILABLE"));
    let text = request(vec![ValueFact::scalar(ValueKindFact::String)]);
    let result = inference::infer(&text, &GPU_DEVICE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.GPU_DEVICE.UNSUPPORTED_ARGUMENT"));
    let row = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(5)]),
        StorageFact::Dense,
    );
    let result = inference::infer(&request(vec![row]), &GPU_DEVICE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.GPU_DEVICE.UNSUPPORTED_ARGUMENT"));
}

#[test]
fn unknown_selector_remains_dynamic_and_valid_index_is_admitted() {
    let unknown = inference::infer(
        &request(vec![ValueFact::unknown(
            runmat_types::DynamicReason::RuntimeValue,
        )]),
        &GPU_DEVICE_CATALOG_ENTRY,
    );
    assert!(unknown.diagnostics.is_empty(), "{:?}", unknown.diagnostics);
    let mut valid = request(vec![ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        },
    ))]);
    valid.literals = LiteralContext::new(vec![LiteralValue::Integer {
        text: "1".into(),
        class: NumericClass::UInt64,
    }]);
    let result = inference::infer(&valid, &GPU_DEVICE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let mut tolerated = request(vec![ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        },
    ))]);
    tolerated.literals = LiteralContext::new(vec![LiteralValue::Number(1.0 - 1e-10)]);
    let result = inference::infer(&tolerated, &GPU_DEVICE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);

    let mut represented_single = request(vec![ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        },
    ))]);
    represented_single.literals = LiteralContext::new(vec![LiteralValue::Real {
        text: "1.00000001".into(),
        class: NumericClass::Single,
    }]);
    let result = inference::infer(&represented_single, &GPU_DEVICE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);

    let mut typed_hex = request(vec![ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        },
    ))]);
    typed_hex.literals = LiteralContext::new(vec![LiteralValue::Integer {
        text: "0x1u64".into(),
        class: NumericClass::UInt64,
    }]);
    let result = inference::infer(&typed_hex, &GPU_DEVICE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);

    typed_hex.literals = LiteralContext::new(vec![LiteralValue::Integer {
        text: "0x2u64".into(),
        class: NumericClass::UInt64,
    }]);
    let result = inference::infer(&typed_hex, &GPU_DEVICE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.GPU_DEVICE.DEVICE_NOT_AVAILABLE"));

    typed_hex.literals = LiteralContext::new(vec![LiteralValue::Integer {
        text: "0xFFFFFFFFFFFFFFFFu64".into(),
        class: NumericClass::UInt64,
    }]);
    let result = inference::infer(&typed_hex, &GPU_DEVICE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.GPU_DEVICE.INVALID_INDEX"));
}

#[test]
fn device_contract_is_host_read_only_and_does_not_claim_kernels() {
    let entry = &GPU_DEVICE_CATALOG_ENTRY;
    assert_eq!(
        entry.contract.async_behavior,
        BuiltinAsyncBehavior::NeverSuspends
    );
    assert_eq!(entry.placement.residency, BuiltinResidencyPolicy::Host);
    assert_eq!(entry.placement.fusion, BuiltinFusionPolicy::Boundary);
    assert_eq!(
        entry.placement.distributed,
        BuiltinDistributedPolicy::Unsupported
    );
    assert!(entry
        .contract
        .effects
        .contains(&runmat_types::EffectKind::EnvironmentRead));
    assert!(entry
        .contract
        .effects
        .contains(&runmat_types::EffectKind::MayThrow));
}
