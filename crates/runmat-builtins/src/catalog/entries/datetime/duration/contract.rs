use super::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

const HOST_EFFECTS: &[EffectKind] = &[EffectKind::MaySuspend, EffectKind::MayThrow];

pub(super) const fn entry(
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
    infer: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    suspends: bool,
    extensions: &'static [BuiltinExtensionDescriptor],
    integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "datetime/duration",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: if suspends {
                BuiltinAsyncBehavior::MaySuspend
            } else {
                BuiltinAsyncBehavior::NeverSuspends
            },
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: if suspends { HOST_EFFECTS } else { &[] },
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            residency: if suspends {
                BuiltinResidencyPolicy::GatherToHost
            } else {
                BuiltinResidencyPolicy::Host
            },
            fusion: BuiltinFusionPolicy::Boundary,
            distributed: BuiltinDistributedPolicy::Unsupported,
        },
        link: BuiltinLinkContract {
            reachability: BuiltinReachability::Always,
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: ExecutionStackRequirement::Any,
            artifact_dependencies: &[],
        },
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions,
        integer_capabilities,
        integer_audit: None,
        suppress_auto_output: false,
    }
}
