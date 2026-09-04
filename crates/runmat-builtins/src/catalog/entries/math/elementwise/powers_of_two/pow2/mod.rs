mod documentation;
mod errors;
mod extensions;
mod integer;
mod signatures;

use crate::*;
use documentation::DOCUMENTATION;
pub use errors::*;
pub use extensions::*;
pub use integer::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};
use signatures::POW2_DESCRIPTOR;

const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];

pub const POW2_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "pow2" },
    category: "math/elementwise",
    documentation: DOCUMENTATION,
    descriptor: &POW2_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::PowerOfTwo(
            PowerOfTwoInferenceRule::Power,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
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
    bindings: &BINDINGS,
    extensions: &POW2_EXTENSIONS,
    integer_capabilities: &POW2_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
