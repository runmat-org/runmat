use crate::*;

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real numeric or RunMat-mode logical scalar, vector, or matrix.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "chr",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host character array with rounded integer text.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "chr = int2str(N)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const INT2STR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INT2STR.INVALID_INPUT",
    identifier: Some("RunMat:int2str:InvalidInput"),
    when: "Input is not a supported real numeric or logical scalar, vector, or matrix.",
    message: "int2str: unsupported input type",
};
pub const INT2STR_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INT2STR.INVALID_OPTION",
    identifier: Some("RunMat:int2str:InvalidOption"),
    when: "Too many arguments were supplied.",
    message: "int2str: invalid option arguments",
};
pub const INT2STR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INT2STR.INTERNAL",
    identifier: Some("RunMat:int2str:InternalError"),
    when: "Character-array assembly or dimensions failed.",
    message: "int2str: internal error",
};
pub const INT2STR_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INT2STR.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:int2str:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "int2str: too many output arguments",
};
pub const INT2STR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        INT2STR_ERROR_INVALID_INPUT,
        INT2STR_ERROR_INVALID_OPTION,
        INT2STR_ERROR_INTERNAL,
        INT2STR_ERROR_TOO_MANY_OUTPUTS,
    ],
};
pub const INT2STR_LOGICAL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "int2str-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "int2str with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Int2strLogicalInputExtension"),
};
pub const INT2STR_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [INT2STR_LOGICAL_EXTENSION];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "N", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are formatted directly from native storage, including wide values above flintmax.",
}];
pub const INT2STR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "chr = int2str(integer_N)", inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "Integer text remains exact and produces a host character array. Resident input is downloaded through its owner before formatting.",
}];
