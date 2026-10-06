//! Catalog authority for timer construction, discovery, methods, and setters.

mod contract;
mod documentation;
mod inference;
mod property;

#[cfg(test)]
mod tests;

pub use contract::{
    TIMERFIND_EXPLICIT_GPU_NUMERIC_FILTER_EXTENSION, TIMER_ERROR_CALLBACK, TIMER_ERROR_GC,
    TIMER_ERROR_INVALID_HANDLE, TIMER_ERROR_INVALID_INPUT, TIMER_ERROR_INVALID_PROPERTY,
    TIMER_EXPLICIT_GPU_NUMERIC_PROPERTY_EXTENSION,
};
pub use property::{TimerBusyMode, TimerExecutionMode, TimerProperty, TimerVisibility};

use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub(crate) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &TIMER,
    &TIMERFIND,
    &TIMERFINDALL,
    &START,
    &STARTAT,
    &STOP,
    &WAIT,
    &DELETE,
    &SET_TIMER_FCN,
    &SET_START_FCN,
    &SET_STOP_FCN,
    &SET_ERROR_FCN,
    &SET_PERIOD,
    &SET_START_DELAY,
    &SET_TASKS_TO_EXECUTE,
    &SET_BUSY_MODE,
    &SET_EXECUTION_MODE,
    &SET_NAME,
    &SET_TAG,
    &SET_OBJECT_VISIBILITY,
    &SET_USER_DATA,
];

macro_rules! timer_effects {
    (TIMER) => {
        &[
            EffectKind::EnvironmentWrite,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    (TIMERFIND) => {
        &[
            EffectKind::EnvironmentRead,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    (TIMERFINDALL) => {
        &[
            EffectKind::EnvironmentRead,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    (START) => {
        &[
            EffectKind::Clock,
            EffectKind::EnvironmentWrite,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    (STARTAT) => {
        &[
            EffectKind::Clock,
            EffectKind::EnvironmentWrite,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    (STOP) => {
        &[
            EffectKind::Clock,
            EffectKind::EnvironmentWrite,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    (WAIT) => {
        &[
            EffectKind::EnvironmentRead,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    (DELETE) => {
        &[
            EffectKind::Clock,
            EffectKind::EnvironmentWrite,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    ($setter:ident) => {
        &[
            EffectKind::EnvironmentWrite,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
}

macro_rules! entry {
    ($id:ident, $name:literal, $doc:expr, $descriptor:expr, $infer:expr,
     $stack:expr, $extensions:expr, $integers:expr, $auto_output:expr) => {
        pub const $id: BuiltinCatalogEntry = BuiltinCatalogEntry {
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "timing",
            documentation: $doc,
            descriptor: $descriptor,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Incomplete,
                inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new($infer)),
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::MaySuspend,
                purity: BuiltinPurity::Impure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: timer_effects!($id),
                // HostRuntime is the current schema's common authority for the
                // session timer, local-task, timing, and source-evaluation ports.
                // Per-port capability identities are not yet representable.
                capabilities: &[CapabilityRequirement::HostRuntime],
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
                execution_stack: $stack,
                artifact_dependencies: &[],
            },
            bindings: &REQUIRED_DEFAULT_BINDING,
            extensions: $extensions,
            integer_capabilities: $integers,
            integer_audit: None,
            suppress_auto_output: $auto_output,
        };
    };
}

entry!(
    TIMER,
    "timer",
    documentation::TIMER,
    &contract::TIMER_DESCRIPTOR,
    inference::constructor,
    ExecutionStackRequirement::Any,
    &contract::TIMER_EXTENSIONS,
    &contract::TIMER_INTEGER_CAPABILITIES,
    false
);
entry!(
    TIMERFIND,
    "timerfind",
    documentation::TIMERFIND,
    &contract::TIMERFIND_DESCRIPTOR,
    inference::find,
    ExecutionStackRequirement::Any,
    &contract::TIMERFIND_EXTENSIONS,
    &contract::TIMERFIND_INTEGER_CAPABILITIES,
    false
);
entry!(
    TIMERFINDALL,
    "timerfindall",
    documentation::TIMERFINDALL,
    &contract::TIMERFINDALL_DESCRIPTOR,
    inference::find,
    ExecutionStackRequirement::Any,
    &contract::TIMERFIND_EXTENSIONS,
    &contract::TIMERFIND_INTEGER_CAPABILITIES,
    false
);

entry!(
    START,
    "__runmat_timer_start",
    documentation::INTERNAL_METHOD,
    &contract::TIMER_METHOD_DESCRIPTOR,
    inference::start,
    ExecutionStackRequirement::Process,
    &[],
    &[],
    true
);
entry!(
    STARTAT,
    "__runmat_timer_startat",
    documentation::INTERNAL_METHOD,
    &contract::TIMER_STARTAT_DESCRIPTOR,
    inference::startat,
    ExecutionStackRequirement::Process,
    &[],
    &[],
    true
);
entry!(
    STOP,
    "__runmat_timer_stop",
    documentation::INTERNAL_METHOD,
    &contract::TIMER_METHOD_DESCRIPTOR,
    inference::stop,
    ExecutionStackRequirement::Process,
    &[],
    &[],
    true
);
entry!(
    WAIT,
    "__runmat_timer_wait",
    documentation::INTERNAL_METHOD,
    &contract::TIMER_METHOD_DESCRIPTOR,
    inference::wait,
    ExecutionStackRequirement::Process,
    &[],
    &[],
    true
);
entry!(
    DELETE,
    "__runmat_timer_delete",
    documentation::INTERNAL_METHOD,
    &contract::TIMER_METHOD_DESCRIPTOR,
    inference::delete,
    ExecutionStackRequirement::Process,
    &[],
    &[],
    true
);

entry!(
    SET_TIMER_FCN,
    "set.TimerFcn",
    documentation::INTERNAL_SETTER,
    &contract::TIMER_SETTER_DESCRIPTOR,
    inference::set_timer_fcn,
    ExecutionStackRequirement::Any,
    &[],
    &[],
    false
);
entry!(
    SET_START_FCN,
    "set.StartFcn",
    documentation::INTERNAL_SETTER,
    &contract::TIMER_SETTER_DESCRIPTOR,
    inference::set_start_fcn,
    ExecutionStackRequirement::Any,
    &[],
    &[],
    false
);
entry!(
    SET_STOP_FCN,
    "set.StopFcn",
    documentation::INTERNAL_SETTER,
    &contract::TIMER_SETTER_DESCRIPTOR,
    inference::set_stop_fcn,
    ExecutionStackRequirement::Any,
    &[],
    &[],
    false
);
entry!(
    SET_ERROR_FCN,
    "set.ErrorFcn",
    documentation::INTERNAL_SETTER,
    &contract::TIMER_SETTER_DESCRIPTOR,
    inference::set_error_fcn,
    ExecutionStackRequirement::Any,
    &[],
    &[],
    false
);
entry!(
    SET_PERIOD,
    "set.Period",
    documentation::INTERNAL_SETTER,
    &contract::TIMER_SETTER_DESCRIPTOR,
    inference::set_period,
    ExecutionStackRequirement::Any,
    &[],
    &[],
    false
);
entry!(
    SET_START_DELAY,
    "set.StartDelay",
    documentation::INTERNAL_SETTER,
    &contract::TIMER_SETTER_DESCRIPTOR,
    inference::set_start_delay,
    ExecutionStackRequirement::Any,
    &[],
    &[],
    false
);
entry!(
    SET_TASKS_TO_EXECUTE,
    "set.TasksToExecute",
    documentation::INTERNAL_SETTER,
    &contract::TIMER_SETTER_DESCRIPTOR,
    inference::set_tasks_to_execute,
    ExecutionStackRequirement::Any,
    &[],
    &[],
    false
);
entry!(
    SET_BUSY_MODE,
    "set.BusyMode",
    documentation::INTERNAL_SETTER,
    &contract::TIMER_SETTER_DESCRIPTOR,
    inference::set_busy_mode,
    ExecutionStackRequirement::Any,
    &[],
    &[],
    false
);
entry!(
    SET_EXECUTION_MODE,
    "set.ExecutionMode",
    documentation::INTERNAL_SETTER,
    &contract::TIMER_SETTER_DESCRIPTOR,
    inference::set_execution_mode,
    ExecutionStackRequirement::Any,
    &[],
    &[],
    false
);
entry!(
    SET_NAME,
    "set.Name",
    documentation::INTERNAL_SETTER,
    &contract::TIMER_SETTER_DESCRIPTOR,
    inference::set_name,
    ExecutionStackRequirement::Any,
    &[],
    &[],
    false
);
entry!(
    SET_TAG,
    "set.Tag",
    documentation::INTERNAL_SETTER,
    &contract::TIMER_SETTER_DESCRIPTOR,
    inference::set_tag,
    ExecutionStackRequirement::Any,
    &[],
    &[],
    false
);
entry!(
    SET_OBJECT_VISIBILITY,
    "set.ObjectVisibility",
    documentation::INTERNAL_SETTER,
    &contract::TIMER_SETTER_DESCRIPTOR,
    inference::set_object_visibility,
    ExecutionStackRequirement::Any,
    &[],
    &[],
    false
);
entry!(
    SET_USER_DATA,
    "set.UserData",
    documentation::INTERNAL_SETTER,
    &contract::TIMER_SETTER_DESCRIPTOR,
    inference::set_user_data,
    ExecutionStackRequirement::Any,
    &[],
    &[],
    false
);
