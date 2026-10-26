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

pub const SIMULANNEALBND_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "simulannealbnd",
    &SIMULANNEALBND_DESCRIPTOR,
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
pub const ACCEPTANCESA_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "acceptancesa",
    &ACCEPTANCESA_DESCRIPTOR,
    documentation::ACCEPTANCESA,
    inference::helper,
    Execution {
        effects: &[
            EffectKind::Randomness,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[],
        stack: ExecutionStackRequirement::Any,
        extensions: &[],
        integers: &[],
        suppress: false,
    },
);
pub const ANNEALINGFAST_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "annealingfast",
    &ANNEALINGFAST_DESCRIPTOR,
    documentation::ANNEALINGFAST,
    inference::helper,
    Execution {
        effects: &[
            EffectKind::Randomness,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[],
        stack: ExecutionStackRequirement::Any,
        extensions: &[],
        integers: &[],
        suppress: false,
    },
);
pub const ANNEALINGBOLTZ_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "annealingboltz",
    &ANNEALINGBOLTZ_DESCRIPTOR,
    documentation::ANNEALINGBOLTZ,
    inference::helper,
    Execution {
        effects: &[
            EffectKind::Randomness,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[],
        stack: ExecutionStackRequirement::Any,
        extensions: &[],
        integers: &[],
        suppress: false,
    },
);
pub const TEMPERATUREEXP_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "temperatureexp",
    &TEMPERATUREEXP_DESCRIPTOR,
    documentation::TEMPERATUREEXP,
    inference::helper,
    Execution {
        effects: &[EffectKind::MaySuspend, EffectKind::MayThrow],
        capabilities: &[],
        stack: ExecutionStackRequirement::Any,
        extensions: &[],
        integers: &[],
        suppress: false,
    },
);
pub const TEMPERATUREFAST_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "temperaturefast",
    &TEMPERATUREFAST_DESCRIPTOR,
    documentation::TEMPERATUREFAST,
    inference::helper,
    Execution {
        effects: &[EffectKind::MaySuspend, EffectKind::MayThrow],
        capabilities: &[],
        stack: ExecutionStackRequirement::Any,
        extensions: &[],
        integers: &[],
        suppress: false,
    },
);
pub const TEMPERATUREBOLTZ_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "temperatureboltz",
    &TEMPERATUREBOLTZ_DESCRIPTOR,
    documentation::TEMPERATUREBOLTZ,
    inference::helper,
    Execution {
        effects: &[EffectKind::MaySuspend, EffectKind::MayThrow],
        capabilities: &[],
        stack: ExecutionStackRequirement::Any,
        extensions: &[],
        integers: &[],
        suppress: false,
    },
);
pub const SAPLOTBESTF_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "saplotbestf",
    &SAPLOTBESTF_DESCRIPTOR,
    documentation::SAPLOTBESTF,
    inference::helper,
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
pub const SAPLOTBESTX_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "saplotbestx",
    &SAPLOTBESTX_DESCRIPTOR,
    documentation::SAPLOTBESTX,
    inference::helper,
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
pub const SAPLOTF_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "saplotf",
    &SAPLOTF_DESCRIPTOR,
    documentation::SAPLOTF,
    inference::helper,
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
pub const SAPLOTX_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "saplotx",
    &SAPLOTX_DESCRIPTOR,
    documentation::SAPLOTX,
    inference::helper,
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
pub const SAPLOTSTOPPING_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "saplotstopping",
    &SAPLOTSTOPPING_DESCRIPTOR,
    documentation::SAPLOTSTOPPING,
    inference::helper,
    Execution {
        effects: &[
            EffectKind::UserInterface,
            EffectKind::Clock,
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
pub const SAPLOTTEMPERATURE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "saplottemperature",
    &SAPLOTTEMPERATURE_DESCRIPTOR,
    documentation::SAPLOTTEMPERATURE,
    inference::helper,
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
