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

/// Iterative methods may invoke arbitrary selected language callbacks. Their
/// host-only numerical route is not an authority to erase callback effects.
pub(super) const fn iterative_entry(spec: FactorEntrySpec) -> BuiltinCatalogEntry {
    let mut entry = host_factor_entry(spec);
    entry.contract.purity = BuiltinPurity::Impure;
    entry.contract.effects = &[
        runmat_types::EffectKind::HostCallback,
        runmat_types::EffectKind::MaySuspend,
        runmat_types::EffectKind::MayThrow,
        runmat_types::EffectKind::Unknown,
    ];
    entry.link.reachability = BuiltinReachability::Dynamic;
    entry.link.execution_stack = runmat_types::ExecutionStackRequirement::Process;
    entry
}

pub(super) fn iterative_effects(
    mut result: runmat_types::CallInference,
    request: &runmat_types::CallRequest,
    callback_positions: &[usize],
    reports_status: bool,
) -> runmat_types::CallInference {
    // A proven numeric operand cannot dispatch a user callback. Unknown
    // operands retain the full conservative authority until selection.
    if callback_positions.iter().all(|&index| {
        request
            .arguments
            .get(index)
            .is_none_or(|fact| matches!(fact.kind, runmat_types::ValueKindFact::Numeric(_)))
    }) {
        result.effects.0.remove(&runmat_types::EffectKind::Unknown);
        if !reports_status
            || request
                .outputs
                .requested
                .known_count()
                .is_some_and(|count| count >= 2)
        {
            result
                .effects
                .0
                .remove(&runmat_types::EffectKind::HostCallback);
        }
    }
    result
}

/// Host-only factors whose exact, pure numeric implementation is admitted for
/// an owned browser worker. No other factor inherits this capability.
pub(super) const fn owned_host_cpu_factor_entry(spec: FactorEntrySpec) -> BuiltinCatalogEntry {
    let mut entry = host_factor_entry(spec);
    entry.bindings = &REQUIRED_OWNED_HOST_CPU_BINDING;
    entry
}
