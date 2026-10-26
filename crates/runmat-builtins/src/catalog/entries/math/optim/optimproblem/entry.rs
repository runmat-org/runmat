use super::*;
use crate::*;
pub const OPTIMPROBLEM_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "optimproblem",
    &OPTIMPROBLEM_DESCRIPTOR,
    documentation::DOCUMENTATION,
    inference::constructor,
);
pub const OPTIMPROBLEM_SUBSREF_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "optim.problemdef.OptimizationProblem.subsref",
    &descriptor::READ,
    documentation::READ,
    inference::read,
);
pub const OPTIMPROBLEM_SUBSASGN_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "optim.problemdef.OptimizationProblem.subsasgn",
    &descriptor::WRITE,
    documentation::WRITE,
    inference::problem,
);
pub(super) const fn entry(
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
