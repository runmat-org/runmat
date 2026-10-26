use super::*;
use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};
const fn entry(
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
    infer: fn(&runmat_types::CallRequest, &BuiltinCatalogEntry) -> runmat_types::CallInference,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "math/optim",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: &[
                EffectKind::MaySuspend,
                EffectKind::MayThrow,
                EffectKind::Unknown,
            ],
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            residency: BuiltinResidencyPolicy::Host,
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
        extensions: &[],
        integer_capabilities: &[],
        integer_audit: None,
        suppress_auto_output: false,
    }
}
pub const OPTIMVAR_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "optimvar",
    &OPTIMVAR_DESCRIPTOR,
    documentation::DOCUMENTATION,
    inference::constructor,
);
pub const OPTIMVAR_SUBSREF_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "optim.problemdef.OptimizationVariable.subsref",
    &OPTIMVAR_SUBSREF_DESCRIPTOR,
    documentation::READ,
    inference::read,
);
pub const OPTIMVAR_SUBSASGN_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "optim.problemdef.OptimizationVariable.subsasgn",
    &OPTIMVAR_SUBSASGN_DESCRIPTOR,
    documentation::WRITE,
    inference::write,
);
