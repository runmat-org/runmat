use super::*;
use crate::*;
const fn entry(
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
    inference: fn(&runmat_types::CallRequest, &BuiltinCatalogEntry) -> runmat_types::CallInference,
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
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: &[
                runmat_types::EffectKind::MaySuspend,
                runmat_types::EffectKind::MayThrow,
                runmat_types::EffectKind::Unknown,
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
            execution_stack: runmat_types::ExecutionStackRequirement::Any,
            artifact_dependencies: &[],
        },
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: &[],
        integer_capabilities: &[],
        integer_audit: None,
        suppress_auto_output: false,
    }
}
pub const OPTIMWARMSTART_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "optimwarmstart",
    &OPTIMWARMSTART_DESCRIPTOR,
    documentation::DOCUMENTATION,
    inference::constructor,
);
pub const LSQLIN_WARM_SUBSREF_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "LsqlinWarmStart.subsref",
    &ACTIVE_WARM_SUBSREF_DESCRIPTOR,
    documentation::READ,
    inference::read,
);
pub const LSQLIN_WARM_SUBSASGN_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "LsqlinWarmStart.subsasgn",
    &ACTIVE_WARM_SUBSASGN_DESCRIPTOR,
    documentation::WRITE,
    inference::write,
);
pub const QUADPROG_WARM_SUBSREF_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "QuadprogWarmStart.subsref",
    &ACTIVE_WARM_SUBSREF_DESCRIPTOR,
    documentation::QUADRATIC_READ,
    inference::read,
);
pub const QUADPROG_WARM_SUBSASGN_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "QuadprogWarmStart.subsasgn",
    &ACTIVE_WARM_SUBSASGN_DESCRIPTOR,
    documentation::QUADRATIC_WRITE,
    inference::write,
);
