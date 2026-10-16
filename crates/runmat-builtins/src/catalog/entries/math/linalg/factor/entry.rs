use crate::*;

const EFFECTS: &[runmat_types::EffectKind] = &[
    runmat_types::EffectKind::MaySuspend,
    runmat_types::EffectKind::MayThrow,
];

pub(super) struct FactorEntrySpec {
    pub name: &'static str,
    pub provenance: BuiltinCatalogProvenance,
    pub documentation: BuiltinDocumentation,
    pub descriptor: &'static BuiltinDescriptor,
    pub inference: IdentityInferenceRule,
    pub extensions: &'static [BuiltinExtensionDescriptor],
    pub integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    pub residency: BuiltinResidencyPolicy,
    pub maturity: BuiltinContractMaturity,
}

pub(super) const fn factor_entry(spec: FactorEntrySpec) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: BuiltinCatalogAudience::Public,
        provenance: spec.provenance,
        identity: BuiltinCatalogIdentity { name: spec.name },
        category: "math/linalg/factor",
        documentation: spec.documentation,
        descriptor: spec.descriptor,
        contract: BuiltinContractDeclaration {
            maturity: spec.maturity,
            inference_rule: BuiltinInferenceRule::Identity(spec.inference),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: EFFECTS,
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Optional,
            residency: spec.residency,
            fusion: BuiltinFusionPolicy::Boundary,
            distributed: BuiltinDistributedPolicy::MaterializeArguments,
        },
        link: BuiltinLinkContract {
            reachability: BuiltinReachability::Always,
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: runmat_types::ExecutionStackRequirement::Any,
            artifact_dependencies: &[],
        },
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: spec.extensions,
        integer_capabilities: spec.integer_capabilities,
        integer_audit: None,
        suppress_auto_output: false,
    }
}

/// The ordinary Schur transformations have no provider factorization hook.
pub(super) const fn host_factor_entry(spec: FactorEntrySpec) -> BuiltinCatalogEntry {
    let mut entry = factor_entry(spec);
    entry.placement.accelerator = BuiltinAcceleratorPolicy::Forbidden;
    entry.placement.distributed = BuiltinDistributedPolicy::Unsupported;
    entry
}

/// Host-only factors whose exact, pure numeric implementation is admitted for
/// an owned browser worker. No other factor inherits this capability.
pub(super) const fn owned_host_cpu_factor_entry(spec: FactorEntrySpec) -> BuiltinCatalogEntry {
    let mut entry = host_factor_entry(spec);
    entry.bindings = &REQUIRED_OWNED_HOST_CPU_BINDING;
    entry
}
