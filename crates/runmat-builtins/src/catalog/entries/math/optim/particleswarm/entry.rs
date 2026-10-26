use super::descriptor::{EXTENSIONS, INTEGER_CAPABILITIES};
use super::*;
use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};
struct Execution {
    effects: &'static [EffectKind],
    capabilities: &'static [runmat_types::CapabilityRequirement],
    stack: ExecutionStackRequirement,
    extensions: &'static [BuiltinExtensionDescriptor],
    integers: &'static [BuiltinIntegerCapabilityDescriptor],
    suppress: bool,
}
const fn entry(
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
    infer: fn(&runmat_types::CallRequest, &BuiltinCatalogEntry) -> runmat_types::CallInference,
    execution: Execution,
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
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: execution.effects,
            capabilities: execution.capabilities,
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            residency: BuiltinResidencyPolicy::GatherToHost,
            fusion: BuiltinFusionPolicy::Boundary,
            distributed: BuiltinDistributedPolicy::Unsupported,
        },
        link: BuiltinLinkContract {
            reachability: BuiltinReachability::Always,
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: execution.stack,
            artifact_dependencies: &[],
        },
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: execution.extensions,
        integer_capabilities: execution.integers,
        integer_audit: None,
        suppress_auto_output: execution.suppress,
    }
}
pub const PARTICLESWARM_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "particleswarm",
    &PARTICLESWARM_DESCRIPTOR,
    documentation::SOLVER,
    inference::solver,
    Execution {
        effects: &[
            EffectKind::FilesystemWrite,
            EffectKind::UserInterface,
            EffectKind::Randomness,
            EffectKind::Clock,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
            EffectKind::Unknown,
        ],
        capabilities: &[],
        stack: ExecutionStackRequirement::Process,
        extensions: EXTENSIONS,
        integers: INTEGER_CAPABILITIES,
        suppress: false,
    },
);
pub const PSWCREATIONUNIFORM_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "pswcreationuniform",
    &PSWCREATIONUNIFORM_DESCRIPTOR,
    documentation::CREATION,
    inference::creation,
    Execution {
        effects: &[
            EffectKind::Randomness,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[],
        stack: ExecutionStackRequirement::Any,
        extensions: EXTENSIONS,
        integers: INTEGER_CAPABILITIES,
        suppress: false,
    },
);
pub const PSWPLOTBESTF_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "pswplotbestf",
    &PSWPLOTBESTF_DESCRIPTOR,
    documentation::PLOT,
    inference::plot,
    Execution {
        effects: &[
            EffectKind::UserInterface,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[runmat_types::CapabilityRequirement::UserInterface],
        stack: ExecutionStackRequirement::Any,
        extensions: &[],
        integers: &[],
        suppress: true,
    },
);
