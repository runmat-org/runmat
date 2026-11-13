use crate::*;

const INTEGER_GAINS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "sys1",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Scalar gain only; the integer rounds at the binary64 model-arithmetic boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "sys2",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Scalar gain only; the integer rounds at the binary64 model-arithmetic boundary.",
    },
];
pub const SERIES_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "sys = series(integer_scalar_gain,sys) or series(sys,integer_scalar_gain)",
    inputs: &INTEGER_GAINS,
    computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::ScalarOnly,
    notes: "A typed integer gain is a RunMat extension; binary64 model-arithmetic rounding precedes transfer-function or state-space composition.",
}];

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "sys1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upstream tf or ss model.",
    },
    BuiltinParamDescriptor {
        name: "sys2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Downstream tf or ss model.",
    },
];
const CHANNEL_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "sys1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upstream tf or ss model.",
    },
    BuiltinParamDescriptor {
        name: "sys2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Downstream tf or ss model.",
    },
    BuiltinParamDescriptor {
        name: "out1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Distinct upstream output indices.",
    },
    BuiltinParamDescriptor {
        name: "in2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Distinct downstream input indices.",
    },
];
const NAMED_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "sys1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upstream model.",
    },
    BuiltinParamDescriptor {
        name: "sys2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Downstream model.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Match uniquely named output and input channels.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cascaded model, with sys1 upstream.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "sys = series(sys1,sys2)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = series(sys1,sys2,out1,in2)",
        inputs: &CHANNEL_INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = series(sys1,sys2,'name')",
        inputs: &NAMED_INPUTS,
        outputs: &OUTPUTS,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.SERIES.INVALID_ARGUMENT",
        identifier: Some("RunMat:series:InvalidArgument"),
        when: "Input or output form is invalid.",
        message: "series: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.SERIES.INVALID_MODEL",
        identifier: Some("RunMat:series:InvalidModel"),
        when: "Model matrices, channels, or sample times are incompatible.",
        message: "series: invalid model",
    },
    BuiltinErrorDescriptor {
        code: "RM.SERIES.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:series:UnsupportedModel"),
        when: "The current model representation cannot express the requested interconnection.",
        message: "series: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.SERIES.INTERNAL",
        identifier: Some("RunMat:series:Internal"),
        when: "Model result construction fails.",
        message: "series: internal error",
    },
];
pub const SERIES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
