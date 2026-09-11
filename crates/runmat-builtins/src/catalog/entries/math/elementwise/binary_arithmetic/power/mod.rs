mod contract;
mod documentation;

#[cfg(test)]
mod inference_tests;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub(super) use contract::INFERENCE_POLICY;
pub use contract::*;

const EFFECTS: &[EffectKind] = &[EffectKind::MaySuspend, EffectKind::MayThrow];

pub const POWER_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "power" },
    category: "math/elementwise",
    documentation: documentation::DOCUMENTATION,
    descriptor: &POWER_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::BinaryArithmetic(
            BinaryArithmeticInferenceRule::Power,
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
    extensions: POWER_EXTENSIONS,
    integer_capabilities: POWER_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const fn power_entry() -> &'static BuiltinCatalogEntry {
    &POWER_CATALOG_ENTRY
}
