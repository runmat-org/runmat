use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCompatibility, BuiltinCompletionPolicy, BuiltinContractDeclaration,
    BuiltinContractMaturity, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinFusionPolicy,
    BuiltinInferenceRule, BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract,
    BuiltinPortability, BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy,
    BuiltinSemanticKind, BuiltinSignatureDescriptor, ParallelInferenceRule,
};
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

const CURRENT_EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
const POOL_READ_EFFECTS: [EffectKind; 3] = [
    EffectKind::EnvironmentRead,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
const POOL_WRITE_EFFECTS: [EffectKind; 4] = [
    EffectKind::EnvironmentRead,
    EffectKind::EnvironmentWrite,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
const SCHEDULE_EFFECTS: [EffectKind; 3] = [
    EffectKind::EnvironmentWrite,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
const FETCH_EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
const PARALLEL_RUNTIME: [CapabilityRequirement; 1] = [CapabilityRequirement::ParallelRuntime];
const CURRENT_PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Host,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: crate::BuiltinDistributedPolicy::Unsupported,
};
const CURRENT_LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};

macro_rules! parallel_entry {
    (
        $constant:ident,
        $name:literal,
        $rule:expr,
        $documentation:expr,
        $descriptor:ident,
        $maturity:expr,
        $async_behavior:expr,
        $purity:expr,
        $effects:expr
    ) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "parallel",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: BuiltinContractDeclaration {
                maturity: $maturity,
                inference_rule: $rule,
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: $async_behavior,
                purity: $purity,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: $effects,
                capabilities: &PARALLEL_RUNTIME,
            },
            placement: CURRENT_PLACEMENT,
            link: CURRENT_LINK,
            bindings: &crate::REQUIRED_DEFAULT_BINDING,
            extensions: &[],
            integer_capabilities: &[],
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

const ANY_REQUIRED: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value operated on by the parallel runtime.",
};
const ANY_OPTIONAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Optional value supplied by the designated lab.",
};
const LAB_REQUIRED: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "lab",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based lab index.",
};
const LAB_OPTIONAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "lab",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Optional one-based source lab index.",
};
const TAG_OPTIONAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tag",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Optional message tag.",
};
const DIMENSION_OPTIONAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dimension",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("2"),
    description: "One-based concatenation dimension.",
};
const REDUCER_REQUIRED: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "reducer",
    ty: BuiltinParamType::Callable,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Associative binary reduction function.",
};
const ANY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result produced by the parallel operation.",
}];

macro_rules! signature {
    ($name:ident, $label:literal, $inputs:expr, $outputs:expr) => {
        const $name: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: $inputs,
            outputs: $outputs,
        }];
    };
}

const LOWERING_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.PARALLEL.LOWERING_REQUIRED",
    identifier: Some("RunMat:parallel:LoweringRequired"),
    when: "The operation is invoked without an active compiler-owned SPMD or distributed execution context.",
    message: "parallel operation requires executor-aware lowering",
}];

macro_rules! descriptor {
    ($name:ident, $signatures:ident) => {
        pub const $name: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &LOWERING_ERRORS,
        };
    };
}

const PARALLEL_EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
const PARALLEL_PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Dynamic,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: crate::BuiltinDistributedPolicy::Unsupported,
};
const DISTRIBUTED_INSPECTION_PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    distributed: crate::BuiltinDistributedPolicy::InspectHandles,
    ..PARALLEL_PLACEMENT
};
const PARALLEL_LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};

macro_rules! documented_parallel_data_entry {
    ($constant:ident, $name:literal, $rule:expr, $documentation:expr, $descriptor:ident) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "parallel",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Complete,
                inference_rule: $rule,
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::MaySuspend,
                purity: BuiltinPurity::Impure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &PARALLEL_EFFECTS,
                capabilities: &PARALLEL_RUNTIME,
            },
            placement: PARALLEL_PLACEMENT,
            link: PARALLEL_LINK,
            bindings: &crate::REQUIRED_DEFAULT_BINDING,
            extensions: &[],
            integer_capabilities: &[],
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

macro_rules! codistributor_entry {
    ($constant:ident, $name:literal, $rule:expr, $documentation:expr, $descriptor:ident) => {
        codistributor_entry!(
            $constant,
            $name,
            $rule,
            $documentation,
            $descriptor,
            PARALLEL_PLACEMENT
        );
    };
    ($constant:ident, $name:literal, $rule:expr, $documentation:expr, $descriptor:ident, $placement:expr) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "parallel",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Complete,
                inference_rule: $rule,
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::NeverSuspends,
                purity: BuiltinPurity::Pure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &[],
                capabilities: &[],
            },
            placement: $placement,
            link: PARALLEL_LINK,
            bindings: &crate::REQUIRED_DEFAULT_BINDING,
            extensions: &[],
            integer_capabilities: &[],
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

mod codistributors;
mod collectives;
mod context;
mod current_execution;
mod distributed_arrays;
mod documentation;
mod futures;
mod pools;

pub use codistributors::*;
pub use collectives::*;
pub use context::*;
pub use current_execution::*;
pub use distributed_arrays::*;
pub use futures::*;
pub use pools::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(
        entries,
        &[
            codistributors::ENTRIES,
            collectives::ENTRIES,
            context::ENTRIES,
            current_execution::ENTRIES,
            distributed_arrays::ENTRIES,
            futures::ENTRIES,
            pools::ENTRIES,
        ],
    );
}
