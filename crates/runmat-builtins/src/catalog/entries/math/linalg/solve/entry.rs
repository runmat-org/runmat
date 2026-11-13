use crate::*;

const EFFECTS: &[runmat_types::EffectKind] = &[
    runmat_types::EffectKind::MaySuspend,
    runmat_types::EffectKind::MayThrow,
];

pub(super) struct SolveEntrySpec {
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

pub(super) const fn solve_entry(spec: SolveEntrySpec) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: BuiltinCatalogAudience::Public,
        provenance: spec.provenance,
        identity: BuiltinCatalogIdentity { name: spec.name },
        category: "math/linalg/solve",
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
