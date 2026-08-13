use super::*;
use crate::{BuiltinAsyncBehavior, BuiltinCompatibility, BuiltinPurity, BuiltinSemanticKind};
use runmat_types::{CapabilityRequirement, EffectKind};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = pilot()",
    inputs: &[],
    outputs: &OUTPUTS,
}];
const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    summary: "Pilot builtin.",
    keywords: &["pilot"],
    related: &[],
    introduced: None,
    status: None,
    examples: &[],
};
const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Optional,
    residency: BuiltinResidencyPolicy::PreserveInputs,
    fusion: BuiltinFusionPolicy::Candidate,
};
const LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    artifact_dependencies: &[],
};
const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
const CAPABILITIES: [CapabilityRequirement; 1] = [CapabilityRequirement::HostRuntime];

const PILOT_ID: BuiltinCatalogIdentity = BuiltinCatalogIdentity { name: "pilot" };
const PILOT_BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    identity: BuiltinBindingIdentity {
        builtin: PILOT_ID,
        variant: "default",
    },
    availability: BuiltinBindingAvailability::Required,
}];
const PILOT: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: PILOT_ID,
    category: "test",
    documentation: DOCUMENTATION,
    descriptor: &DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRuleId("test.pilot"),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &CAPABILITIES,
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: &PILOT_BINDINGS,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: false,
};

const SECOND_ID: BuiltinCatalogIdentity = BuiltinCatalogIdentity { name: "second" };
const SECOND_BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    identity: BuiltinBindingIdentity {
        builtin: SECOND_ID,
        variant: "default",
    },
    availability: BuiltinBindingAvailability::Required,
}];
const SECOND: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: SECOND_ID,
    bindings: &SECOND_BINDINGS,
    ..PILOT
};

#[test]
fn valid_catalog_has_stable_order_independent_fingerprint() {
    assert!(validate_builtin_catalog(&[&PILOT, &SECOND]).is_empty());
    assert_eq!(
        canonical_catalog_fingerprint(&[&PILOT, &SECOND]).unwrap(),
        canonical_catalog_fingerprint(&[&SECOND, &PILOT]).unwrap()
    );
}

#[test]
fn validation_rejects_duplicate_catalog_and_binding_identities() {
    let errors = validate_builtin_catalog(&[&PILOT, &PILOT]);
    assert!(errors
        .iter()
        .any(|error| error.message == "duplicate builtin catalog identity"));
    assert!(errors
        .iter()
        .any(|error| error.message == "duplicate builtin binding identity"));
}

#[test]
fn migrated_registry_is_valid_and_case_insensitive() {
    assert!(validate_builtin_catalog(builtin_catalog_entries()).is_empty());
    assert_eq!(
        builtin_catalog_entry_by_name("FULL")
            .expect("full catalog entry")
            .contract
            .inference_rule
            .0,
        "array.full"
    );
}

#[test]
fn full_contract_densifies_sparse_facts_without_losing_class_shape_or_residency() {
    use runmat_types::{
        CallRequest, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };

    let mut input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt32,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Sparse,
    );
    input.residency = runmat_types::ResidencyFact::Host;
    let request = CallRequest {
        arguments: vec![input.clone()],
        literals: runmat_types::LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("full").expect("full entry"),
        &request,
    );
    assert!(inference.diagnostics.is_empty());
    assert_eq!(inference.outputs[0].kind, input.kind);
    assert_eq!(inference.outputs[0].shape, input.shape);
    assert_eq!(inference.outputs[0].residency, input.residency);
    assert_eq!(inference.outputs[0].storage, StorageFact::Dense);
}
