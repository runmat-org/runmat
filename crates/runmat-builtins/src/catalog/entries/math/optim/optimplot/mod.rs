//! Real predefined optimization chart callbacks sharing one typed kind domain.
mod descriptor;
mod documentation;
mod inference;
mod kind;
mod surrogate;
#[cfg(test)]
mod tests;
use crate::*;
pub use descriptor::*;
pub use kind::OptimPlotKind;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};
pub use surrogate::{
    SURROGATEOPTPLOT_CATALOG_ENTRY, SURROGATEOPTPLOT_DESCRIPTOR,
    SURROGATEOPTPLOT_ERROR_INVALID_ARGUMENT,
};
const fn entry(
    kind: OptimPlotKind,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name: kind.name() },
        category: "math/optim",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Complete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
                inference::infer,
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
            capabilities: &[CapabilityRequirement::UserInterface],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Optional,
            residency: BuiltinResidencyPolicy::Dynamic,
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
pub const OPTIMPLOTX_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    OptimPlotKind::CurrentPoint,
    &OPTIMPLOTX_DESCRIPTOR,
    documentation::X,
);
pub const OPTIMPLOTFUNCCOUNT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    OptimPlotKind::FunctionCount,
    &OPTIMPLOTFUNCCOUNT_DESCRIPTOR,
    documentation::COUNT,
);
pub const OPTIMPLOTFVAL_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    OptimPlotKind::FunctionValue,
    &OPTIMPLOTFVAL_DESCRIPTOR,
    documentation::VALUE,
);
pub const OPTIMPLOTCONSTRVIOLATION_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    OptimPlotKind::ConstraintViolation,
    &OPTIMPLOTCONSTRVIOLATION_DESCRIPTOR,
    documentation::CONSTRAINT,
);
pub const OPTIMPLOTSTEPSIZE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    OptimPlotKind::StepSize,
    &OPTIMPLOTSTEPSIZE_DESCRIPTOR,
    documentation::STEP,
);
pub const OPTIMPLOTFIRSTORDEROPT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    OptimPlotKind::FirstOrderOptimality,
    &OPTIMPLOTFIRSTORDEROPT_DESCRIPTOR,
    documentation::FIRSTORDER,
);
pub const OPTIMPLOTFVALCONSTR_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    OptimPlotKind::ConstrainedFunctionValue,
    &OPTIMPLOTFVALCONSTR_DESCRIPTOR,
    documentation::FEASIBLE,
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &OPTIMPLOTX_CATALOG_ENTRY,
    &OPTIMPLOTFUNCCOUNT_CATALOG_ENTRY,
    &OPTIMPLOTFVAL_CATALOG_ENTRY,
    &OPTIMPLOTCONSTRVIOLATION_CATALOG_ENTRY,
    &OPTIMPLOTSTEPSIZE_CATALOG_ENTRY,
    &OPTIMPLOTFIRSTORDEROPT_CATALOG_ENTRY,
    &OPTIMPLOTFVALCONSTR_CATALOG_ENTRY,
    &SURROGATEOPTPLOT_CATALOG_ENTRY,
];
