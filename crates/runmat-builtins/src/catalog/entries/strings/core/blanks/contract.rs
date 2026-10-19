use crate::*;
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of space characters.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-row character array of spaces.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "s = blanks(n)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const BLANKS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLANKS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:blanks:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "blanks: too many output arguments",
};
pub const BLANKS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[BLANKS_ERROR_TOO_MANY_OUTPUTS],
};
pub const BLANKS_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "blanks-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "blanks with a GPU-resident length is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BlanksGpuInputExtension"),
};
pub const BLANKS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [BLANKS_GPU_INPUT_EXTENSION];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "n", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "n accepts integer-valued single, double, and all eight native integer classes. Negative values act as zero.",
}];
pub const BLANKS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "s = blanks(n)", inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::NotApplicable,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::StructuralParameter,
    notes: "n controls the exact host character-array width; oversized counts reject. A resident scalar is available only under the RunMat extension gate.",
}];
