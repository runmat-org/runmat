//! Canonical timer identity signatures, errors, integer capabilities, and extensions.

use crate::*;

pub const TIMER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Timer handle object.",
}];

pub const TIMER_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Name,Value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Timer properties such as TimerFcn, StartDelay, Period, ExecutionMode, Name, Tag, and UserData.",
}];

pub const TIMER_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
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

pub const TIMERFIND_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "timers",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matching timer handle, empty cell row, or cell row of timer handles.",
}];

pub const TIMERFIND_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Name,Value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Property filters matched by exact value.",
}];

pub const TIMERFIND_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
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

pub const TIMERFINDALL_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
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

pub const TIMER_METHOD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Zero on success.",
}];

pub const TIMER_METHOD_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Timer handle or cell array of timer handles.",
}];

pub const TIMER_METHOD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "method(t)",
        inputs: &TIMER_METHOD_INPUTS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = method(t) (RunMat extension)",
        inputs: &TIMER_METHOD_INPUTS,
        outputs: &TIMER_METHOD_OUTPUT,
    },
];

pub const TIMER_STARTAT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "t",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Timer handle or cell array of timer handles.",
    },
    BuiltinParamDescriptor {
        name: "delay",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Positive relative delay in seconds.",
    },
];
pub const TIMER_STARTAT_ABSOLUTE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "t",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Timer handle or timer cell array.",
    },
    BuiltinParamDescriptor {
        name: "firingTime",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Absolute datetime, date text, date vector, or serial-date array.",
    },
];
pub const TIMER_STARTAT_COMPONENT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "t",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Timer handle or timer cell array.",
    },
    BuiltinParamDescriptor {
        name: "year, month, day[, hour, minute, second]",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Three or six broadcast-compatible calendar components.",
    },
];
pub const TIMER_STARTAT_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "startat(t, firingTime)",
        inputs: &TIMER_STARTAT_ABSOLUTE_INPUTS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "startat(t, year, month, day[, hour, minute, second])",
        inputs: &TIMER_STARTAT_COMPONENT_INPUTS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "startat(t, relativeSeconds) (RunMat extension)",
        inputs: &TIMER_STARTAT_INPUTS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = startat(t, relativeSeconds) (RunMat extension)",
        inputs: &TIMER_STARTAT_INPUTS,
        outputs: &TIMER_METHOD_OUTPUT,
    },
];

pub const TIMER_SETTER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated timer object.",
}];

pub const TIMER_SETTER_INPUTS: [BuiltinParamDescriptor; 2] = [
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

pub const TIMER_SETTER_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = set.Property(obj, value)",
    inputs: &TIMER_SETTER_INPUTS,
    outputs: &TIMER_SETTER_OUTPUT,
}];

pub const TIMER_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMER.INVALID_INPUT",
    identifier: Some("RunMat:timer:InvalidInput"),
    when: "A timer argument, property, or callback value has an unsupported type.",
    message: "timer: invalid input",
};

pub const TIMER_ERROR_INVALID_PROPERTY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMER.INVALID_PROPERTY",
    identifier: Some("RunMat:timer:InvalidProperty"),
    when: "A timer property name is unknown, read-only, or has an invalid value.",
    message: "timer: invalid property",
};

pub const TIMER_ERROR_INVALID_HANDLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMER.INVALID_HANDLE",
    identifier: Some("RunMat:timer:InvalidHandle"),
    when: "A lifecycle method receives a non-timer or invalid timer handle.",
    message: "timer: invalid timer handle",
};

pub const TIMER_ERROR_CALLBACK: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMER.CALLBACK",
    identifier: Some("RunMat:timer:CallbackFailed"),
    when: "A timer callback fails.",
    message: "timer: callback failed",
};

pub const TIMER_ERROR_GC: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMER.GC",
    identifier: Some("RunMat:timer:GcFailure"),
    when: "A timer object cannot be allocated, rooted, read, or mutated.",
    message: "timer: object storage failed",
};

pub const TIMER_ERRORS: [BuiltinErrorDescriptor; 5] = [
    TIMER_ERROR_INVALID_INPUT,
    TIMER_ERROR_INVALID_PROPERTY,
    TIMER_ERROR_INVALID_HANDLE,
    TIMER_ERROR_CALLBACK,
    TIMER_ERROR_GC,
];

pub const TIMER_EXPLICIT_GPU_NUMERIC_PROPERTY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "timer-explicit-gpu-numeric-property",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "timer with an explicitly GPU-resident numeric property is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TimerExplicitGpuNumericPropertyExtension"),
    };
pub const TIMER_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [TIMER_EXPLICIT_GPU_NUMERIC_PROPERTY_EXTENSION];
pub const TIMER_METHOD_STATUS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "timer-method-status-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "assigning the numeric status result of start, startat, stop, wait, or delete is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TimerMethodStatusOutputExtension"),
};
pub const TIMER_METHOD_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [TIMER_METHOD_STATUS_EXTENSION];
pub const TIMER_STARTAT_RELATIVE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "timer-startat-relative-seconds",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "startat with omitted time or relative seconds is a RunMat extension; MATLAB specifies an absolute firing time",
    error_identifier: Some("RunMat:compatibility:TimerStartatRelativeExtension"),
};
pub const TIMER_STARTAT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    TIMER_STARTAT_RELATIVE_EXTENSION,
    TIMER_METHOD_STATUS_EXTENSION,
];
pub const TIMERFIND_EXPLICIT_GPU_NUMERIC_FILTER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "timerfind-explicit-gpu-numeric-filter",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "timerfind or timerfindall with an explicitly GPU-resident numeric filter is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TimerfindExplicitGpuNumericFilterExtension"),
    };
pub const TIMERFIND_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [TIMERFIND_EXPLICIT_GPU_NUMERIC_FILTER_EXTENSION];

pub const TIMER_INTEGER_PROPERTY_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "Period, StartDelay, or TasksToExecute",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Numeric timer properties accept native integer scalars. TasksToExecute remains an exact structural integer; duration properties cross one checked seconds boundary.",
    },
];
pub const TIMER_INTEGER_USER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "UserData",
        classes: &ALL_INTEGER_CLASSES,
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

pub const TIMERFIND_INTEGER_FILTER_INPUT: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "property value",
        classes: &ALL_INTEGER_CLASSES,
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

pub const TIMER_STARTAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIMER_STARTAT_SIGNATURES,
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
