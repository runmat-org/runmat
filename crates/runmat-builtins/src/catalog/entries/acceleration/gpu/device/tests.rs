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
fn device_is_a_host_typed_handle_with_read_only_provider_property_types() {
    for entry in ENTRIES {
        assert_eq!(
            entry.placement.accelerator,
            BuiltinAcceleratorPolicy::Forbidden
        );
        assert_eq!(entry.placement.residency, BuiltinResidencyPolicy::Host);
        assert!(entry
            .contract
            .capabilities
            .contains(&CapabilityRequirement::Accelerator));
    }
    let result = inference::infer(&request(vec![]), &GPU_DEVICE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    let ValueKindFact::Object(fields) = &result.outputs[0].kind else {
        panic!("gpuDevice should infer an object")
    };
    assert!(!fields.properties_complete);
    assert!(fields.runtime_class.as_ref().unwrap().is(GPU_DEVICE_CLASS));
    assert_eq!(fields.handle_semantics, Some(true));
    let known = &fields.properties;
    for name in [
        "Index",
        "Name",
        "TotalMemory",
        "AvailableMemory",
        "SupportsDouble",
        "DeviceAvailable",
        "DeviceSelected",
    ] {
        assert!(known.contains_key(name), "missing {name}");
    }
    assert!(!known.contains_key("backend"));
    assert!(!known.contains_key("memory_bytes"));
    assert!(matches!(&known["Name"].kind, ValueKindFact::Character));
    assert!(
        matches!(&known["Index"].kind, ValueKindFact::Numeric(fact) if fact.class == NumericClass::Double)
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
fn selector_empty_deselects_and_availability_remains_a_runtime_fact() {
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
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape.element_count(), Some(0));
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
    assert!(
        result.diagnostics.is_empty(),
        "device count is not a static literal 1"
    );
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
    let mut fractional = request(vec![ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        },
    ))]);
    for value in [1.0 - 1e-10, 1.0 + 1e-10] {
        fractional.literals = LiteralContext::new(vec![LiteralValue::Number(value)]);
        let result = inference::infer(&fractional, &GPU_DEVICE_CATALOG_ENTRY);
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == GPU_DEVICE_ERROR_INVALID_INDEX.code));
    }

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
    assert!(
        result.diagnostics.is_empty(),
        "availability is checked against actual leased owners"
    );

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
fn device_contract_is_host_stateful_suspending_and_does_not_claim_kernels() {
    let entry = &GPU_DEVICE_CATALOG_ENTRY;
    assert_eq!(
        entry.contract.async_behavior,
        BuiltinAsyncBehavior::MaySuspend
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
    assert!(entry
        .contract
        .effects
        .contains(&runmat_types::EffectKind::EnvironmentWrite));
    assert_eq!(entry.contract.maturity, BuiltinContractMaturity::Incomplete);
    assert_eq!(
        entry.extensions.len(),
        1,
        "the provider-info facade gate is retired"
    );
}

#[test]
fn executable_members_have_typed_inference_and_actual_effects() {
    let device = inference::infer(&request(vec![]), &GPU_DEVICE_CATALOG_ENTRY)
        .outputs
        .remove(0);
    // All property names/bindings come from the closed class-member enum.
    assert_eq!(GpuDeviceProperty::ALL.len(), 7);
    assert!(matches!(
        GpuDeviceProperty::Name.fact().kind,
        ValueKindFact::Character
    ));
    assert!(matches!(
        GpuDeviceProperty::DeviceSelected.fact().kind,
        ValueKindFact::Logical
    ));
    assert!(matches!(device.kind, ValueKindFact::Object(_)));
    assert_eq!(
        methods::RESET.contract.async_behavior,
        BuiltinAsyncBehavior::MaySuspend
    );
    assert_eq!(
        methods::NAME.contract.async_behavior,
        BuiltinAsyncBehavior::NeverSuspends
    );
    assert_eq!(
        methods::RESET.descriptor.completion_policy,
        BuiltinCompletionPolicy::MethodOnly
    );
}
