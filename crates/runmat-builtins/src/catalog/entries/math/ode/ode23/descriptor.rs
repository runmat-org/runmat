use crate::*;

const ODE23_OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Solution states evaluated over tspan.",
}];

const ODE23_OUTPUT_TY: [BuiltinParamDescriptor; 2] = [
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

const ODE23_INPUTS_CORE: [BuiltinParamDescriptor; 3] = [
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

const ODE23_INPUTS_WITH_OPTIONS: [BuiltinParamDescriptor; 4] = [
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

const ODE23_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "y = ode23(odefun, tspan, y0)",
        inputs: &ODE23_INPUTS_CORE,
        outputs: &ODE23_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = ode23(odefun, tspan, y0, options)",
        inputs: &ODE23_INPUTS_WITH_OPTIONS,
        outputs: &ODE23_OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[t, y] = ode23(odefun, tspan, y0)",
        inputs: &ODE23_INPUTS_CORE,
        outputs: &ODE23_OUTPUT_TY,
    },
    BuiltinSignatureDescriptor {
        label: "[t, y] = ode23(odefun, tspan, y0, options)",
        inputs: &ODE23_INPUTS_WITH_OPTIONS,
        outputs: &ODE23_OUTPUT_TY,
    },
];

pub const ODE23_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ODE23.INVALID_ARGUMENT",
    identifier: Some("RunMat:ode23:InvalidArgument"),
    when: "Input argument count/options struct grammar is invalid.",
    message: "ode23: invalid argument",
};

pub const ODE23_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ODE23.INVALID_INPUT",
    identifier: Some("RunMat:ode23:InvalidInput"),
    when: "ODE input/state/callback semantics are invalid for integration.",
    message: "ode23: invalid input",
};

pub const ODE23_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ODE23.INTERNAL",
    identifier: Some("RunMat:ode23:Internal"),
    when: "Internal output materialization fails.",
    message: "ode23: internal runtime failure",
};

const ODE23_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ODE23_ERROR_INVALID_ARGUMENT,
    ODE23_ERROR_INVALID_INPUT,
    ODE23_ERROR_INTERNAL,
];

pub const ODE23_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ODE23_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ODE23_ERRORS,
};
