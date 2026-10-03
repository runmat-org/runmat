use super::*;
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, OutputSelection, RequestedOutputCount,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
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
    let result = inference::device(&request(vec![]), &GPU_DEVICE_CATALOG_ENTRY);
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
    let many = inference::device(
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
    let nonscalar = inference::device(&request(vec![matrix]), &GPU_DEVICE_CATALOG_ENTRY);
    assert!(nonscalar
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.GPU_DEVICE.INVALID_INDEX"));
}

#[test]
fn info_is_host_string_even_when_provider_is_absent() {
    let result = inference::info(&request(vec![]), &GPU_INFO_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].kind, ValueKindFact::String);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    let extra = inference::info(
        &request(vec![ValueFact::scalar(ValueKindFact::Unknown)]),
        &GPU_INFO_CATALOG_ENTRY,
    );
    assert!(extra
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.GPU_INFO.TOO_MANY_INPUTS"));
}

#[test]
fn metadata_contracts_do_not_claim_gpu_work_or_suspension() {
    for entry in [&GPU_DEVICE_CATALOG_ENTRY, &GPU_INFO_CATALOG_ENTRY] {
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
    }
    assert!(GPU_DEVICE_CATALOG_ENTRY
        .contract
        .effects
        .contains(&runmat_types::EffectKind::MayThrow));
    assert!(!GPU_INFO_CATALOG_ENTRY
        .contract
        .effects
        .contains(&runmat_types::EffectKind::MayThrow));
}
