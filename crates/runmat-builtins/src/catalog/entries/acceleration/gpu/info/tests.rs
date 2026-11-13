use super::*;
use runmat_types::{
    CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ResidencyFact, ValueFact,
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
fn info_is_host_string_even_when_provider_is_absent() {
    let result = inference::infer(&request(vec![]), &GPU_INFO_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].kind, ValueKindFact::String);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    let extra = inference::infer(
        &request(vec![ValueFact::scalar(ValueKindFact::Unknown)]),
        &GPU_INFO_CATALOG_ENTRY,
    );
    assert!(extra
        .diagnostics
        .iter()
        .any(|d| d.code == GPU_INFO_ERROR_TOO_MANY_INPUTS.code));
    assert!(GPU_INFO_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == GPU_INFO_ERROR_TOO_MANY_INPUTS.code));
}

#[test]
fn info_has_no_integer_form_and_no_gpu_kernel_route() {
    let entry = &GPU_INFO_CATALOG_ENTRY;
    assert_eq!(
        entry.integer_audit.unwrap().kind,
        BuiltinIntegerAuditKind::NotApplicable
    );
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
