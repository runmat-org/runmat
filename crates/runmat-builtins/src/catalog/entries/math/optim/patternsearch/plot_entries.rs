use super::*;
use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};
const fn entry(
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "math/optim",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::DynamicByDesign,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
                inference::plot,
            )),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: &[
                EffectKind::UserInterface,
                EffectKind::HostCallback,
                EffectKind::MaySuspend,
                EffectKind::MayThrow,
            ],
            capabilities: &[runmat_types::CapabilityRequirement::UserInterface],
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
        suppress_auto_output: true,
    }
}
pub const PSPLOTBESTF_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "psplotbestf",
    &PSPLOTBESTF_DESCRIPTOR,
    plot_documentation::BESTF,
);
pub const PSPLOTFUNCOUNT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "psplotfuncount",
    &PSPLOTFUNCOUNT_DESCRIPTOR,
    plot_documentation::COUNT,
);
pub const PSPLOTMESHSIZE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "psplotmeshsize",
    &PSPLOTMESHSIZE_DESCRIPTOR,
    plot_documentation::MESH,
);
pub const PSPLOTBESTX_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "psplotbestx",
    &PSPLOTBESTX_DESCRIPTOR,
    plot_documentation::POINT,
);
pub const PSPLOTMAXCONSTR_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "psplotmaxconstr",
    &PSPLOTMAXCONSTR_DESCRIPTOR,
    plot_documentation::CONSTRAINT,
);
