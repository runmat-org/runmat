use crate::*;
use runmat_types::{CallInference, CallRequest};
pub(super) struct Definition {
    pub name: &'static str,
    pub provenance: BuiltinCatalogProvenance,
    pub documentation: BuiltinDocumentation,
    pub descriptor: &'static BuiltinDescriptor,
    pub inference: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    pub extensions: &'static [BuiltinExtensionDescriptor],
    pub integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
}
pub(super) const fn entry(def: Definition) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: BuiltinCatalogAudience::Public,
        provenance: def.provenance,
        identity: BuiltinCatalogIdentity { name: def.name },
        category: "table",
        documentation: def.documentation,
        descriptor: def.descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
                def.inference,
            )),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: &[
                runmat_types::EffectKind::MaySuspend,
                runmat_types::EffectKind::MayThrow,
            ],
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            residency: BuiltinResidencyPolicy::GatherToHost,
            fusion: BuiltinFusionPolicy::Boundary,
            distributed: BuiltinDistributedPolicy::Unsupported,
        },
        link: BuiltinLinkContract {
            reachability: BuiltinReachability::Always,
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: runmat_types::ExecutionStackRequirement::Any,
            artifact_dependencies: &[],
        },
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: def.extensions,
        integer_capabilities: def.integer_capabilities,
        integer_audit: None,
        suppress_auto_output: false,
    }
}
