use crate::*;
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf", ty: BuiltinParamType::LogicalArray, arity: BuiltinParamArity::Required,
    default: None, description: "Host logical validity mask for numeric handle arrays, or a scalar logical for other candidates.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Graphics receiver object, numeric handle alias, or another value to test.",
}];
const TYPED_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor { name: "h", ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required, default: None,
        description: "Graphics receiver object, numeric handle alias, or another value to test." },
    BuiltinParamDescriptor { name: "type", ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required, default: None,
        description: "Case-insensitive graphics type name as scalar string or character row; unknown names match nothing." },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "tf = isgraphics(h)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = isgraphics(h, type)",
        inputs: &TYPED_INPUTS,
        outputs: &OUTPUT,
    },
];
pub const ISGRAPHICS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISGRAPHICS.INVALID_ARGUMENT",
    identifier: Some("RunMat:isgraphics:InvalidArgument"),
    when: "The input count is not one or two, or the optional type is not host scalar text.",
    message: "isgraphics: invalid argument",
};
pub const ISGRAPHICS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISGRAPHICS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:isgraphics:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "isgraphics: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    ISGRAPHICS_ERROR_INVALID_ARGUMENT,
    ISGRAPHICS_ERROR_TOO_MANY_OUTPUTS,
];
pub const ISGRAPHICS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const ISGRAPHICS_INTEGER_HANDLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "isgraphics-integer-handle-alias",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "isgraphics numeric integer handle aliases are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IsgraphicsIntegerHandleExtension"),
    };
pub const ISGRAPHICS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [ISGRAPHICS_INTEGER_HANDLE_EXTENSION];
const INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "h", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Host integer aliases are read from authoritative native storage and must convert exactly to the existing f64 graphics encoding. Disabled extensions and nonrepresentable aliases return false, not an extension error.",
}];
pub const ISGRAPHICS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "tf = isgraphics(integer_handle_alias, type?)",
    inputs: &INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::Predicate,
    output_class: BuiltinIntegerOutputClassRule::Logical, overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
    notes: "Numeric tensors retain their full shape, including low rank and empty axes. Integer scalars return scalar logical. Resident values are metadata-only false masks regardless of element class; no gather or provider work occurs.",
}];
