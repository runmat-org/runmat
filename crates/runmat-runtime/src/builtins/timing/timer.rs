//! MATLAB-compatible timer handle objects.
//!
//! Timer objects are handle objects with MATLAB timer properties. The runtime
//! session owns discovery and lifecycle state; the host's local task driver
//! polls callbacks after `start` returns.

pub(crate) mod property;
mod arguments;
mod lifecycle;
use arguments::{apply_name_value_pairs, prepare_timer_property_pairs, PreparedTimerPair};
use lifecycle::{start_one_timer, stop_one_timer, wait_one_timer};
use property::{TimerBusyMode, TimerExecutionMode, TimerProperty, TimerVisibility};

use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_types::MemberAccess;
use runmat_value::NumericScalar;
use std::collections::HashMap;

#[cfg(all(test, not(target_arch = "wasm32")))]
use once_cell::sync::Lazy;
#[cfg(all(test, not(target_arch = "wasm32")))]
use std::sync::Mutex;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, HandleRef, IntValue, ObjectInstance, StructValue, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const TIMER_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("timer");
const BUILTIN_TIMER: &str = "timer";
const TIMER_CALLABLE_START: &str = "__runmat_timer_start";
const TIMER_CALLABLE_STARTAT: &str = "__runmat_timer_startat";
const TIMER_CALLABLE_STOP: &str = "__runmat_timer_stop";
const TIMER_CALLABLE_WAIT: &str = "__runmat_timer_wait";
const TIMER_CALLABLE_DELETE: &str = "__runmat_timer_delete";

const CALLBACK_PROPS: [&str; 4] = ["TimerFcn", "StartFcn", "StopFcn", "ErrorFcn"];

#[cfg(all(test, not(target_arch = "wasm32")))]
static TIMER_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

const TIMER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Timer handle object.",
}];

const TIMER_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Name,Value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Timer properties such as TimerFcn, StartDelay, Period, ExecutionMode, Name, Tag, and UserData.",
}];

const TIMER_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "t = timer",
        inputs: &[],
        outputs: &TIMER_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "t = timer(Name, Value, ...)",
        inputs: &TIMER_INPUTS,
        outputs: &TIMER_OUTPUT,
    },
];

const TIMERFIND_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "timers",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matching timer handle, empty cell row, or cell row of timer handles.",
}];

const TIMERFIND_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Name,Value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Property filters matched by exact value.",
}];

const TIMERFIND_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "timers = timerfind",
        inputs: &[],
        outputs: &TIMERFIND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "timers = timerfind(Name, Value, ...)",
        inputs: &TIMERFIND_INPUTS,
        outputs: &TIMERFIND_OUTPUT,
    },
];

const TIMERFINDALL_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "timers = timerfindall",
        inputs: &[],
        outputs: &TIMERFIND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "timers = timerfindall(Name, Value, ...)",
        inputs: &TIMERFIND_INPUTS,
        outputs: &TIMERFIND_OUTPUT,
    },
];

const TIMER_METHOD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Zero on success.",
}];

const TIMER_METHOD_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Timer handle or cell array of timer handles.",
}];

const TIMER_METHOD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "status = method(t)",
    inputs: &TIMER_METHOD_INPUTS,
    outputs: &TIMER_METHOD_OUTPUT,
}];

const TIMER_SETTER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated timer object.",
}];

const TIMER_SETTER_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Timer object receiver.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Property value to validate and assign.",
    },
];

const TIMER_SETTER_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = set.Property(obj, value)",
    inputs: &TIMER_SETTER_INPUTS,
    outputs: &TIMER_SETTER_OUTPUT,
}];

const TIMER_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMER.INVALID_INPUT",
    identifier: Some("RunMat:timer:InvalidInput"),
    when: "A timer argument, property, or callback value has an unsupported type.",
    message: "timer: invalid input",
};

const TIMER_ERROR_INVALID_PROPERTY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMER.INVALID_PROPERTY",
    identifier: Some("RunMat:timer:InvalidProperty"),
    when: "A timer property name is unknown, read-only, or has an invalid value.",
    message: "timer: invalid property",
};

const TIMER_ERROR_INVALID_HANDLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMER.INVALID_HANDLE",
    identifier: Some("RunMat:timer:InvalidHandle"),
    when: "A lifecycle method receives a non-timer or invalid timer handle.",
    message: "timer: invalid timer handle",
};

const TIMER_ERROR_CALLBACK: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMER.CALLBACK",
    identifier: Some("RunMat:timer:CallbackFailed"),
    when: "A timer callback fails.",
    message: "timer: callback failed",
};

const TIMER_ERROR_GC: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMER.GC",
    identifier: Some("RunMat:timer:GcFailure"),
    when: "A timer object cannot be allocated, rooted, read, or mutated.",
    message: "timer: object storage failed",
};

const TIMER_ERRORS: [BuiltinErrorDescriptor; 5] = [
    TIMER_ERROR_INVALID_INPUT,
    TIMER_ERROR_INVALID_PROPERTY,
    TIMER_ERROR_INVALID_HANDLE,
    TIMER_ERROR_CALLBACK,
    TIMER_ERROR_GC,
];

const TIMER_EXPLICIT_GPU_NUMERIC_PROPERTY_EXTENSION: runmat_builtins::BuiltinExtensionDescriptor =
    runmat_builtins::BuiltinExtensionDescriptor {
        id: "timer-explicit-gpu-numeric-property",
        mode: runmat_builtins::BuiltinExtensionMode::RunMatOnly,
        description: "timer with an explicitly GPU-resident numeric property is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TimerExplicitGpuNumericPropertyExtension"),
    };
pub const TIMER_EXTENSIONS: [runmat_builtins::BuiltinExtensionDescriptor; 1] =
    [TIMER_EXPLICIT_GPU_NUMERIC_PROPERTY_EXTENSION];
const TIMERFIND_EXPLICIT_GPU_NUMERIC_FILTER_EXTENSION: runmat_builtins::BuiltinExtensionDescriptor =
    runmat_builtins::BuiltinExtensionDescriptor {
        id: "timerfind-explicit-gpu-numeric-filter",
        mode: runmat_builtins::BuiltinExtensionMode::RunMatOnly,
        description: "timerfind or timerfindall with an explicitly GPU-resident numeric filter is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TimerfindExplicitGpuNumericFilterExtension"),
    };
pub const TIMERFIND_EXTENSIONS: [runmat_builtins::BuiltinExtensionDescriptor; 1] =
    [TIMERFIND_EXPLICIT_GPU_NUMERIC_FILTER_EXTENSION];

const TIMER_INTEGER_PROPERTY_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "Period, StartDelay, or TasksToExecute",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Numeric timer properties accept native integer scalars. TasksToExecute remains an exact structural integer; duration properties cross one checked seconds boundary.",
    },
];
const TIMER_INTEGER_USER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "UserData",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "UserData accepts arbitrary values and preserves native integer class, shape, and payload without numeric conversion.",
    },
];
pub const TIMER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "t = timer(..., numeric_property, integer_value, ...)",
        inputs: &TIMER_INTEGER_PROPERTY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "TasksToExecute is decoded directly from authoritative integer storage. Period and StartDelay require an exactly representable binary64 seconds value.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "t = timer(..., \"UserData\", integer_value, ...)",
        inputs: &TIMER_INTEGER_USER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The timer object preserves the original integer class and value.",
    },
];

const TIMERFIND_INTEGER_FILTER_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "property value",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Property filters compare integer values exactly, including signed values and values wider than binary64's exact integer range.",
    },
];
pub const TIMERFIND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "timers = timerfind[all](..., property, integer_value, ...)",
        inputs: &TIMERFIND_INTEGER_FILTER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Native integer filters are normalized to an exact scalar representation and never routed through f64.",
    },
];

pub const TIMER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIMER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TIMER_ERRORS,
};

pub const TIMERFIND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIMERFIND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TIMER_ERRORS,
};

pub const TIMERFINDALL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIMERFINDALL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TIMER_ERRORS,
};

pub const TIMER_METHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIMER_METHOD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &TIMER_ERRORS,
};

pub const TIMER_SETTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIMER_SETTER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &TIMER_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::timing::timer")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "timer",
    op_kind: GpuOpKind::Custom("timer"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Timer handles are host control-flow objects. Timer callbacks may call GPU-capable builtins, but timer itself has no provider kernel.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::timing::timer")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "timer",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Timer object creation, discovery, and callback execution are side-effect boundaries and are excluded from fusion.",
};

#[runtime_builtin(
    name = "timer",
    category = "timing",
    summary = "Create a timer handle object for scheduled callbacks.",
    keywords = "timer,callback,start,stop,wait,timerfind",
    descriptor(crate::builtins::timing::timer::TIMER_DESCRIPTOR),
    extensions(crate::builtins::timing::timer::TIMER_EXTENSIONS),
    integer_capabilities(crate::builtins::timing::timer::TIMER_INTEGER_CAPABILITIES),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_builtin(
    runtime: Option<RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    ensure_timer_class_registered();
    let args = prepare_timer_property_pairs(&args, false).await?;
    let name = runtime
        .timer_session()
        .borrow_mut()
        .next_name()
        .ok_or_else(|| timer_error(&TIMER_ERROR_INVALID_INPUT, "timer: name identity exhausted"))?;
    let mut object = default_timer_object(name);
    apply_name_value_pairs(&mut object, &args, true)?;
    let Some(Value::String(visibility_text)) = object.properties.get("ObjectVisibility") else {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: invalid ObjectVisibility value",
        ));
    };
    let visibility = TimerVisibility::parse(visibility_text).ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: invalid ObjectVisibility value",
        )
    })?;
    let target = runmat_gc::gc_allocate(Value::Object(object))
        .map_err(|err| timer_error(&TIMER_ERROR_GC, format!("timer: {err}")))?;
    runmat_gc::gc_add_root(target)
        .map_err(|err| timer_error(&TIMER_ERROR_GC, format!("timer: {err}")))?;
    let handle = HandleRef {
        class_name: TIMER_CLASS.into(),
        target,
        valid: true,
    };
    runtime
        .timer_session()
        .borrow_mut()
        .register(handle.clone(), visibility);
    Ok(Value::HandleObject(handle))
}

#[runtime_builtin(
    name = "timerfind",
    category = "timing",
    summary = "Find visible timer objects by property value.",
    keywords = "timerfind,timer,callback,handle",
    descriptor(crate::builtins::timing::timer::TIMERFIND_DESCRIPTOR),
    extensions(crate::builtins::timing::timer::TIMERFIND_EXTENSIONS),
    integer_capabilities(crate::builtins::timing::timer::TIMERFIND_INTEGER_CAPABILITIES),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timerfind_builtin(
    runtime: Option<RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    find_timers(
        &runtime,
        prepare_timer_property_pairs(&args, true).await?,
        false,
    )
}

#[runtime_builtin(
    name = "timerfindall",
    category = "timing",
    summary = "Find all timer objects by property value.",
    keywords = "timerfindall,timer,callback,handle",
    descriptor(crate::builtins::timing::timer::TIMERFINDALL_DESCRIPTOR),
    extensions(crate::builtins::timing::timer::TIMERFIND_EXTENSIONS),
    integer_capabilities(crate::builtins::timing::timer::TIMERFIND_INTEGER_CAPABILITIES),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timerfindall_builtin(
    runtime: Option<RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    find_timers(
        &runtime,
        prepare_timer_property_pairs(&args, true).await?,
        true,
    )
}

#[runtime_builtin(
    name = "__runmat_timer_start",
    execution_stack = "process",
    category = "timing",
    summary = "Start a timer object.",
    keywords = "timer,start",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::timing::timer::TIMER_METHOD_DESCRIPTOR),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_start_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    start_timer_values(&runtime, &value, None).await?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "__runmat_timer_startat",
    execution_stack = "process",
    category = "timing",
    summary = "Start a timer object at a requested time.",
    keywords = "timer,startat",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::timing::timer::TIMER_METHOD_DESCRIPTOR),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_startat_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    let prepared = match rest.first() {
        Some(delay) => {
            prepare_timer_property_pairs(
                &[Value::String("StartDelay".to_string()), delay.clone()],
                false,
            )
            .await?
        }
        None => Vec::new(),
    };
    let delay = prepared
        .first()
        .map(|pair| numeric_scalar(&pair.value, "startat"))
        .transpose()?
        .filter(|seconds| *seconds > 0.0);
    start_timer_values(&runtime, &value, delay).await?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "__runmat_timer_stop",
    execution_stack = "process",
    category = "timing",
    summary = "Stop a timer object.",
    keywords = "timer,stop",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::timing::timer::TIMER_METHOD_DESCRIPTOR),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_stop_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    stop_timer_values(&runtime, &value, true).await?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "__runmat_timer_wait",
    execution_stack = "process",
    category = "timing",
    summary = "Wait until timer objects finish.",
    keywords = "timer,wait",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::timing::timer::TIMER_METHOD_DESCRIPTOR),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_wait_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    wait_timer_values(&runtime, &value).await?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "__runmat_timer_delete",
    execution_stack = "process",
    category = "timing",
    summary = "Delete a timer object.",
    keywords = "timer,delete",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::timing::timer::TIMER_METHOD_DESCRIPTOR),
    context = true,
    builtin_path = "crate::builtins::timing::timer"
)]
pub async fn timer_delete_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    let runtime = timer_runtime(runtime)?;
    delete_timer_values(&runtime, &value).await?;
    Ok(Value::Num(0.0))
}

macro_rules! timer_setter_builtin {
    ($fn_name:ident, $builtin_name:literal, $property:literal) => {
        #[runtime_builtin(
                            name = $builtin_name,
                            category = "timing",
                            summary = "Validate and assign a timer property.",
                            keywords = "timer,set,property",
                            descriptor(crate::builtins::timing::timer::TIMER_SETTER_DESCRIPTOR),
                            builtin_path = "crate::builtins::timing::timer"
                        )]
        pub async fn $fn_name(obj: Value, value: Value) -> BuiltinResult<Value> {
            set_timer_property_object(obj, $property, value).await
        }
    };
}

timer_setter_builtin!(timer_set_timer_fcn_builtin, "set.TimerFcn", "TimerFcn");
timer_setter_builtin!(timer_set_start_fcn_builtin, "set.StartFcn", "StartFcn");
timer_setter_builtin!(timer_set_stop_fcn_builtin, "set.StopFcn", "StopFcn");
timer_setter_builtin!(timer_set_error_fcn_builtin, "set.ErrorFcn", "ErrorFcn");
timer_setter_builtin!(timer_set_period_builtin, "set.Period", "Period");
timer_setter_builtin!(
    timer_set_start_delay_builtin,
    "set.StartDelay",
    "StartDelay"
);
timer_setter_builtin!(
    timer_set_tasks_to_execute_builtin,
    "set.TasksToExecute",
    "TasksToExecute"
);
timer_setter_builtin!(timer_set_busy_mode_builtin, "set.BusyMode", "BusyMode");
timer_setter_builtin!(
    timer_set_execution_mode_builtin,
    "set.ExecutionMode",
    "ExecutionMode"
);
timer_setter_builtin!(timer_set_name_builtin, "set.Name", "Name");
timer_setter_builtin!(timer_set_tag_builtin, "set.Tag", "Tag");
timer_setter_builtin!(
    timer_set_object_visibility_builtin,
    "set.ObjectVisibility",
    "ObjectVisibility"
);
timer_setter_builtin!(timer_set_user_data_builtin, "set.UserData", "UserData");

fn ensure_timer_class_registered() {
    if crate::class_registry::get_class(&TIMER_CLASS.owned()).is_some() {
        return;
    }

    let mut methods = HashMap::new();
    for (name, function_name) in [
        (
            runmat_types::StaticMethodName::new("start"),
            TIMER_CALLABLE_START,
        ),
        (
            runmat_types::StaticMethodName::new("startat"),
            TIMER_CALLABLE_STARTAT,
        ),
        (
            runmat_types::StaticMethodName::new("stop"),
            TIMER_CALLABLE_STOP,
        ),
        (
            runmat_types::StaticMethodName::new("wait"),
            TIMER_CALLABLE_WAIT,
        ),
        (
            runmat_types::StaticMethodName::new("delete"),
            TIMER_CALLABLE_DELETE,
        ),
    ] {
        methods.insert(
            name.into(),
            crate::class_registry::RuntimeMethod {
                name: name.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: function_name.into(),
                implicit_class_argument: None,
            },
        );
    }

    let mut properties = HashMap::new();
    for property in TimerProperty::ALL {
        let name = property.name();
        properties.insert(
            name.into(),
            crate::class_registry::RuntimeProperty {
                name: name.into(),
                is_static: false,
                is_constant: false,
                is_dependent: !property.is_readonly(),
                get_access: MemberAccess::Public,
                set_access: if property.is_readonly() {
                    MemberAccess::Private
                } else {
                    MemberAccess::Public
                },
                default_value: None,
            },
        );
    }

    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: TIMER_CLASS.into(),
        parent: Some(runmat_types::standard::HANDLE.into()),
        properties,
        methods,
    });
}

fn default_timer_object(name: String) -> ObjectInstance {
    let mut object = ObjectInstance::new(TIMER_CLASS.to_string());
    for prop in CALLBACK_PROPS {
        object
            .properties
            .insert(prop.to_string(), Value::String(String::new()));
    }
    object
        .properties
        .insert("Period".to_string(), Value::Num(1.0));
    object
        .properties
        .insert("StartDelay".to_string(), Value::Num(0.0));
    object
        .properties
        .insert("TasksToExecute".to_string(), Value::Num(1.0));
    object
        .properties
        .insert("BusyMode".to_string(), Value::String("drop".to_string()));
    object.properties.insert(
        "ExecutionMode".to_string(),
        Value::String("singleShot".to_string()),
    );
    object
        .properties
        .insert("Name".to_string(), Value::String(name));
    object
        .properties
        .insert("Tag".to_string(), Value::String(String::new()));
    object.properties.insert(
        "ObjectVisibility".to_string(),
        Value::String("on".to_string()),
    );
    object
        .properties
        .insert("UserData".to_string(), Value::Num(0.0));
    object
        .properties
        .insert("AveragePeriod".to_string(), Value::Num(f64::NAN));
    object
        .properties
        .insert("InstantPeriod".to_string(), Value::Num(f64::NAN));
    object
        .properties
        .insert("Running".to_string(), Value::String("off".to_string()));
    object
        .properties
        .insert("TasksExecuted".to_string(), Value::Num(0.0));
    object
        .properties
        .insert("Type".to_string(), Value::String(TIMER_CLASS.to_string()));
    object.properties.insert(
        crate::HANDLE_VALID_FLAG_PROPERTY.to_string(),
        Value::Bool(true),
    );
    object
}

async fn set_timer_property_object(
    obj: Value,
    property_name: &'static str,
    value: Value,
) -> BuiltinResult<Value> {
    let Value::Object(mut object) = obj else {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!("timer: set.{property_name} requires timer object receiver"),
        ));
    };
    if !object.class_name.is(TIMER_CLASS) {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!(
                "timer: set.{property_name} requires timer object receiver, got '{}'",
                object.class_name
            ),
        ));
    }
    let args =
        prepare_timer_property_pairs(&[Value::String(property_name.to_string()), value], false)
            .await?;
    apply_name_value_pairs(&mut object, &args, false)?;
    Ok(Value::Object(object))
}

fn canonical_property_name(name: &str) -> BuiltinResult<TimerProperty> {
    let trimmed = name.trim();
    TimerProperty::parse(trimmed).ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!("timer: unknown property '{trimmed}'"),
        )
    })
}

fn normalize_property_value(name: TimerProperty, value: Value) -> BuiltinResult<Value> {
    match name {
        TimerProperty::Period => {
            let seconds = numeric_scalar(&value, name.name())?;
            if !seconds.is_finite() || seconds <= 0.001 {
                return Err(timer_error(
                    &TIMER_ERROR_INVALID_PROPERTY,
                    "timer: Period must be a finite scalar greater than 0.001",
                ));
            }
            Ok(Value::Num(seconds))
        }
        TimerProperty::StartDelay => {
            let seconds = numeric_scalar(&value, name.name())?;
            if !seconds.is_finite() || seconds < 0.0 {
                return Err(timer_error(
                    &TIMER_ERROR_INVALID_PROPERTY,
                    "timer: StartDelay must be a finite non-negative scalar",
                ));
            }
            Ok(Value::Num(seconds))
        }
        TimerProperty::TasksToExecute => {
            parse_tasks_to_execute_value(&value)?;
            if let Some(integer) = tensor::scalar_integer_value(&value) {
                Ok(Value::Int(integer))
            } else {
                Ok(Value::Num(numeric_scalar(&value, name.name())?.round()))
            }
        }
        TimerProperty::BusyMode => {
            parse_choice(&value, name, TimerBusyMode::parse, TimerBusyMode::spelling)
        }
        TimerProperty::ExecutionMode => parse_choice(
            &value,
            name,
            TimerExecutionMode::parse,
            TimerExecutionMode::spelling,
        ),
        TimerProperty::ObjectVisibility => parse_choice(
            &value,
            name,
            TimerVisibility::parse,
            TimerVisibility::spelling,
        ),
        TimerProperty::Name | TimerProperty::Tag => Ok(Value::String(value_to_string(&value)?)),
        TimerProperty::TimerFcn
        | TimerProperty::StartFcn
        | TimerProperty::StopFcn
        | TimerProperty::ErrorFcn => normalize_callback(value, name.name()),
        TimerProperty::UserData => Ok(value),
        _ => Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!("timer: property '{}' cannot be assigned", name.name()),
        )),
    }
}

fn normalize_callback(value: Value, name: &str) -> BuiltinResult<Value> {
    match value {
        Value::String(_)
        | Value::CharArray(runmat_value::CharArray { rows: 1, .. })
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Ok(value),
        Value::Cell(ref cell)
            if cell.data.is_empty()
                || cell.data.first().is_some_and(is_callback_function) => Ok(value),
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!(
                "timer: {name} must be text, a function handle, or a callback cell array, got {other:?}"
            ),
        )),
    }
}

fn is_callback_function(value: &Value) -> bool {
    matches!(
        value,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    )
}

fn parse_choice<T: Copy>(
    value: &Value,
    name: TimerProperty,
    parse: fn(&str) -> Option<T>,
    spelling: fn(T) -> &'static str,
) -> BuiltinResult<Value> {
    let text = value_to_string(value)?;
    parse(&text)
        .map(|choice| Value::String(spelling(choice).to_string()))
        .ok_or_else(|| {
            timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                format!("timer: invalid {} value '{text}'", name.name()),
            )
        })
}

fn value_to_string(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            format!("timer: expected string scalar or character row, got {other:?}"),
        )),
    }
}

fn numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        let exact = match integer {
            IntValue::I8(value) => i128::from(value),
            IntValue::I16(value) => i128::from(value),
            IntValue::I32(value) => i128::from(value),
            IntValue::I64(value) => i128::from(value),
            IntValue::U8(value) => i128::from(value),
            IntValue::U16(value) => i128::from(value),
            IntValue::U32(value) => i128::from(value),
            IntValue::U64(value) => i128::from(value),
        };
        const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
        if !(-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&exact) {
            return Err(timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                format!("timer: {name} must be exactly representable as double seconds"),
            ));
        }
        return Ok(exact as f64);
    }
    match value {
        Value::Num(value) => Ok(*value),
        Value::Tensor(value) if tensor::is_scalar_tensor(value) => {
            Ok(tensor::tensor_value_f64(value, 0))
        }
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            format!("timer: {name} must be a numeric scalar, got {other:?}"),
        )),
    }
}

fn parse_tasks_to_execute_value(value: &Value) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return parse_tasks_to_execute_integer(&integer);
    }

    let count = numeric_scalar(value, "TasksToExecute")?;
    if !count.is_finite() || count < 1.0 || count.fract() != 0.0 {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: TasksToExecute must be a positive integer scalar",
        ));
    }
    if !fits_platform_usize(count) {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: TasksToExecute exceeds platform limits",
        ));
    }
    Ok(count as usize)
}

fn parse_tasks_to_execute_integer(value: &IntValue) -> BuiltinResult<usize> {
    let count = value.try_to_usize().ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: TasksToExecute must be a positive integer scalar",
        )
    })?;
    if count == 0 {
        return Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: TasksToExecute must be a positive integer scalar",
        ));
    }
    Ok(count)
}

fn fits_platform_usize(value: f64) -> bool {
    value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64)
}

async fn start_timer_values(
    runtime: &RuntimeContext,
    value: &Value,
    override_delay: Option<f64>,
) -> BuiltinResult<()> {
    match value {
        Value::HandleObject(handle) if is_timer_handle(handle) => {
            start_one_timer(runtime, handle.clone(), override_delay).await
        }
        Value::Cell(cell) => {
            for item in &cell.data {
                Box::pin(start_timer_values(runtime, item, override_delay)).await?;
            }
            Ok(())
        }
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!("timer: expected timer handle, got {other:?}"),
        )),
    }
}

async fn stop_timer_values(
    runtime: &RuntimeContext,
    value: &Value,
    call_stop: bool,
) -> BuiltinResult<()> {
    match value {
        Value::HandleObject(handle) if is_timer_handle(handle) => {
            stop_one_timer(runtime, handle, call_stop).await
        }
        Value::Cell(cell) => {
            for item in &cell.data {
                Box::pin(stop_timer_values(runtime, item, call_stop)).await?;
            }
            Ok(())
        }
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!("timer: expected timer handle, got {other:?}"),
        )),
    }
}

async fn wait_timer_values(runtime: &RuntimeContext, value: &Value) -> BuiltinResult<()> {
    match value {
        Value::HandleObject(handle) if is_timer_handle(handle) => {
            wait_one_timer(runtime, handle).await
        }
        Value::Cell(cell) => {
            for item in &cell.data {
                Box::pin(wait_timer_values(runtime, item)).await?;
            }
            Ok(())
        }
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!("timer: expected timer handle, got {other:?}"),
        )),
    }
}

async fn delete_timer_values(runtime: &RuntimeContext, value: &Value) -> BuiltinResult<()> {
    match value {
        Value::HandleObject(handle) if is_timer_handle(handle) => {
            stop_one_timer(runtime, handle, false).await?;
            crate::set_handle_valid(handle, false);
            let pending = runtime.timer_session().borrow_mut().delete(handle);
            if let Some(lease) = pending {
                lease.cancel();
            }
            let _ = runmat_gc::gc_remove_root(handle.target);
            Ok(())
        }
        Value::Cell(cell) => {
            for item in &cell.data {
                Box::pin(delete_timer_values(runtime, item)).await?;
            }
            Ok(())
        }
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            format!("timer: expected timer handle, got {other:?}"),
        )),
    }
}

fn update_average_period(handle: &HandleRef, instant_period: f64, idx: usize) -> BuiltinResult<()> {
    let current = numeric_property(handle, "AveragePeriod").unwrap_or(f64::NAN);
    let average = if current.is_finite() {
        (current * (idx as f64 - 1.0) + instant_period) / idx as f64
    } else {
        instant_period
    };
    set_property(handle, "AveragePeriod", Value::Num(average))
}

async fn run_callback(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    property_name: &str,
    event_type: &str,
) -> BuiltinResult<()> {
    let callback = property(handle, property_name)?;
    if callback_is_empty(&callback) {
        return Ok(());
    }
    match callback {
        Value::String(text) if !text.trim().is_empty() => {
            return Err(timer_error(
                &TIMER_ERROR_CALLBACK,
                "timer: text callback execution requires dynamic-eval callback infrastructure; use a function handle callback",
            ));
        }
        Value::CharArray(chars) if !chars.data.is_empty() => {
            let text: String = chars.data.iter().collect();
            if !text.trim().is_empty() {
                return Err(timer_error(
                    &TIMER_ERROR_CALLBACK,
                    "timer: text callback execution requires dynamic-eval callback infrastructure; use a function handle callback",
                ));
            }
        }
        Value::Cell(cell) => {
            let Some(function) = cell.data.first().cloned() else {
                return Ok(());
            };
            let mut args = callback_base_args(runtime, handle, event_type);
            args.extend(cell.data.iter().skip(1).cloned());
            crate::call_feval_async_with_outputs_in_context(runtime, function, &args, 0).await?;
        }
        function @ (Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)) => {
            let args = callback_base_args(runtime, handle, event_type);
            crate::call_feval_async_with_outputs_in_context(runtime, function, &args, 0).await?;
        }
        _ => {}
    }
    Ok(())
}

fn callback_base_args(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    event_type: &str,
) -> Vec<Value> {
    vec![
        Value::HandleObject(handle.clone()),
        timer_event(runtime, event_type),
    ]
}

fn timer_event(runtime: &RuntimeContext, event_type: &str) -> Value {
    let mut data = StructValue::new();
    data.insert(
        "time",
        Value::Num(
            runtime
                .service_ports()
                .timing()
                .wall_time_since_epoch()
                .as_secs_f64(),
        ),
    );
    let mut event = StructValue::new();
    event.insert("Type", Value::String(event_type.to_string()));
    event.insert("Data", Value::Struct(data));
    Value::Struct(event)
}

fn callback_is_empty(value: &Value) -> bool {
    match value {
        Value::String(text) => text.trim().is_empty(),
        Value::CharArray(chars) => chars.data.iter().collect::<String>().trim().is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        _ => false,
    }
}

fn find_timers(
    runtime: &RuntimeContext,
    args: Vec<PreparedTimerPair>,
    include_invisible: bool,
) -> BuiltinResult<Value> {
    ensure_timer_class_registered();
    let filters: Vec<_> = args
        .into_iter()
        .map(|pair| (pair.property, normalize_find_value(pair.value)))
        .collect();
    let mut matches = Vec::new();
    let handles = runtime.timer_session().borrow().handles();
    for handle in handles
        .iter()
        .filter(|handle| crate::is_handle_valid(handle))
    {
        let visibility = string_property(handle, "ObjectVisibility")
            .ok()
            .and_then(|value| TimerVisibility::parse(&value));
        if let Some(visibility) = visibility {
            if let Some(entry) = runtime.timer_session().borrow_mut().record_mut(handle) {
                entry.visibility = visibility;
            }
        }
        let admitted_visibility = runtime
            .timer_session()
            .borrow()
            .record(handle)
            .map(|entry| entry.visibility);
        if !include_invisible && admitted_visibility != Some(TimerVisibility::On) {
            continue;
        }
        if filters_match(handle, &filters).unwrap_or(false) {
            matches.push(Value::HandleObject(handle.clone()));
        }
    }
    handles_to_result(matches)
}

fn normalize_find_value(value: Value) -> Value {
    if let Some(integer) = tensor::scalar_integer_value(&value) {
        return Value::Int(integer);
    }
    match value {
        Value::CharArray(chars) if chars.rows == 1 => Value::String(chars.data.iter().collect()),
        other => other,
    }
}

fn filters_match(handle: &HandleRef, filters: &[(TimerProperty, Value)]) -> BuiltinResult<bool> {
    for (name, expected) in filters {
        let actual = normalize_find_value(property(handle, name.name())?);
        if !timer_values_equal(&actual, expected) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn timer_values_equal(lhs: &Value, rhs: &Value) -> bool {
    match (lhs, rhs) {
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Num(a), Value::Num(b)) if a.is_nan() && b.is_nan() => true,
        (Value::Num(a), Value::Num(b)) => a == b,
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Int(a), Value::Num(b)) | (Value::Num(b), Value::Int(a)) => {
            crate::builtins::logical::rel::integer_comparison::compare_numeric_scalars_exact(
                NumericScalar::from(a.clone()),
                NumericScalar::F64(*b),
            ) == Some(std::cmp::Ordering::Equal)
        }
        (Value::Bool(a), Value::Bool(b)) => a == b,
        _ => lhs == rhs,
    }
}

fn handles_to_result(handles: Vec<Value>) -> BuiltinResult<Value> {
    if handles.len() == 1 {
        Ok(handles.into_iter().next().unwrap())
    } else {
        let len = handles.len();
        CellArray::new(handles, 1, len)
            .map(Value::Cell)
            .map_err(|err| timer_error(&TIMER_ERROR_INVALID_INPUT, err))
    }
}

fn property(handle: &HandleRef, name: &str) -> BuiltinResult<Value> {
    ensure_valid_timer(handle)?;
    runmat_gc::gc_with_value(&handle.target, |target| {
        let Value::Object(object) = target else {
            return None;
        };
        object.properties.get(name).cloned()
    })
    .map_err(|err| timer_error(&TIMER_ERROR_GC, format!("timer: {err}")))?
    .ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!("timer: missing property '{name}'"),
        )
    })
}

fn set_property(handle: &HandleRef, name: &str, value: Value) -> BuiltinResult<()> {
    ensure_valid_timer(handle)?;
    let updated = runmat_gc::gc_with_value_mut(&handle.target, |target| {
        let Value::Object(object) = target else {
            return false;
        };
        object.properties.insert(name.to_string(), value);
        true
    })
    .map_err(|err| timer_error(&TIMER_ERROR_GC, format!("timer: {err}")))?;
    if updated {
        Ok(())
    } else {
        Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            "timer: handle target is not an object",
        ))
    }
}

fn string_property(handle: &HandleRef, name: &str) -> BuiltinResult<String> {
    match property(handle, name)? {
        Value::String(text) => Ok(text),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        other => Err(timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!("timer: property '{name}' must be text, got {other:?}"),
        )),
    }
}

fn numeric_property(handle: &HandleRef, name: &str) -> BuiltinResult<f64> {
    numeric_scalar(&property(handle, name)?, name)
}

fn running_is_on(object: &ObjectInstance) -> bool {
    matches!(
        object.properties.get("Running"),
        Some(Value::String(text)) if text.eq_ignore_ascii_case("on")
    )
}

fn is_timer_handle(handle: &HandleRef) -> bool {
    handle.class_name == TIMER_CLASS
}

fn ensure_valid_timer(handle: &HandleRef) -> BuiltinResult<()> {
    if is_timer_handle(handle) && crate::is_handle_valid(handle) {
        Ok(())
    } else {
        Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            "timer: invalid or deleted timer handle",
        ))
    }
}

fn timer_error(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        error.message.to_string()
    } else {
        detail.to_string()
    };
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_TIMER);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn timer_runtime(runtime: Option<RuntimeContext>) -> BuiltinResult<RuntimeContext> {
    runtime.ok_or_else(|| {
        timer_error(
            &TIMER_ERROR_INVALID_INPUT,
            "timer: runtime session is required for timer ownership",
        )
    })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod browser_tests;
