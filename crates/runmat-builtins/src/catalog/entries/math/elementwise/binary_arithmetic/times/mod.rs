mod contract;
mod documentation;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub use contract::*;

const EFFECTS: &[EffectKind] = &[EffectKind::MaySuspend, EffectKind::MayThrow];

pub const TIMES_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "times" },
    category: "math/elementwise",
    documentation: documentation::DOCUMENTATION,
    descriptor: &TIMES_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::BinaryArithmetic(
            BinaryArithmeticInferenceRule::Multiply,
        )),
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
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Candidate,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: TIMES_EXTENSIONS,
    integer_capabilities: TIMES_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const fn times_entry() -> &'static BuiltinCatalogEntry {
    &TIMES_CATALOG_ENTRY
}
