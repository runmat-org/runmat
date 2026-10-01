use crate::*;

const ODE15S_OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Solution states evaluated over tspan.",
}];

const ODE15S_OUTPUT_TY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "t",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Time points selected by solver.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Solution states at each returned time point.",
    },
];

const ODE15S_INPUTS_CORE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "odefun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "ODE right-hand-side callback f(t,y).",
    },
    BuiltinParamDescriptor {
        name: "tspan",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Time interval or monotonic time vector.",
    },
    BuiltinParamDescriptor {
        name: "y0",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Initial state vector/value.",
    },
];

const ODE15S_INPUTS_WITH_OPTIONS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "odefun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "ODE right-hand-side callback f(t,y).",
    },
    BuiltinParamDescriptor {
        name: "tspan",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Time interval or monotonic time vector.",
    },
    BuiltinParamDescriptor {
        name: "y0",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Initial state vector/value.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional struct with tolerances and step controls.",
    },
];

const ODE15S_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "y = ode15s(odefun, tspan, y0)",
        inputs: &ODE15S_INPUTS_CORE,
        outputs: &ODE15S_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = ode15s(odefun, tspan, y0, options)",
        inputs: &ODE15S_INPUTS_WITH_OPTIONS,
        outputs: &ODE15S_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[t, y] = ode15s(odefun, tspan, y0)",
        inputs: &ODE15S_INPUTS_CORE,
        outputs: &ODE15S_OUTPUT_TY,
    },
    BuiltinSignatureDescriptor {
        label: "[t, y] = ode15s(odefun, tspan, y0, options)",
        inputs: &ODE15S_INPUTS_WITH_OPTIONS,
        outputs: &ODE15S_OUTPUT_TY,
    },
];

pub const ODE15S_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ODE15S.INVALID_ARGUMENT",
    identifier: Some("RunMat:ode15s:InvalidArgument"),
    when: "Input argument count/options struct grammar is invalid.",
    message: "ode15s: invalid argument",
};

pub const ODE15S_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ODE15S.INVALID_INPUT",
    identifier: Some("RunMat:ode15s:InvalidInput"),
    when: "ODE input/state/callback semantics are invalid for integration.",
    message: "ode15s: invalid input",
};

pub const ODE15S_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ODE15S.INTERNAL",
    identifier: Some("RunMat:ode15s:Internal"),
    when: "Internal output materialization fails.",
    message: "ode15s: internal runtime failure",
};

const ODE15S_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ODE15S_ERROR_INVALID_ARGUMENT,
    ODE15S_ERROR_INVALID_INPUT,
    ODE15S_ERROR_INTERNAL,
];

pub const ODE15S_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ODE15S_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ODE15S_ERRORS,
};
