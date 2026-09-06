mod contract;
mod documentation;
pub(super) mod inference;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub use contract::*;

const EFFECTS: &[EffectKind] = &[EffectKind::EnvironmentRead, EffectKind::MayThrow];

pub const ISENV_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "isenv" },
    category: "io/repl_fs",
    documentation: documentation::DOCUMENTATION,
    descriptor: &ISENV_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Io(IoInferenceRule::ReplFs(
            IoReplFsInferenceRule::Environment(EnvironmentInferenceRule::Exists),
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::DeterministicReadOnly,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
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
    integer_audit: Some(&ISENV_INTEGER_AUDIT),
    suppress_auto_output: false,
};
