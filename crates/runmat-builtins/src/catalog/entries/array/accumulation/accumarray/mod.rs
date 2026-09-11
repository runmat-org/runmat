mod documentation;
mod errors;
mod integer;
mod signatures;

use crate::*;
use documentation::DOCUMENTATION;
pub use errors::*;
pub use integer::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};
use signatures::ACCUMARRAY_DESCRIPTOR;

const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 3] = [
    EffectKind::HostCallback,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub const ACCUMARRAY_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "accumarray" },
    category: "array/accumulation",
    documentation: DOCUMENTATION,
    descriptor: &ACCUMARRAY_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Array(ArrayInferenceRule::Accumulation(
            AccumulationInferenceRule::Indexed,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Never,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Process,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &[],
    integer_capabilities: &ACCUMARRAY_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
