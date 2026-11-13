use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ALL_INTEGER_CLASSES,
};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Uniform scalar callback results in the singleton-expansion shape.",
}];
const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Binary function handle; callable text is a RunMat-only extension.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left scalar or array input.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right scalar or array input.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "C = bsxfun(fun, A, B)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const BSXFUN_ERROR_INVALID_FUNCTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BSXFUN.INVALID_FUNCTION",
    identifier: Some("RunMat:bsxfun:InvalidFunction"),
    when: "The first input cannot be used as a binary function.",
    message: "bsxfun: first input must be a function handle or callable name",
};
pub const BSXFUN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BSXFUN.INVALID_INPUT",
    identifier: Some("RunMat:bsxfun:InvalidInput"),
    when: "An input cannot supply scalar numeric, logical, complex, or character values.",
    message: "bsxfun: inputs must be numeric, logical, complex, or character arrays",
};
pub const BSXFUN_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BSXFUN.SIZE_MISMATCH",
    identifier: Some("RunMat:bsxfun:SizeMismatch"),
    when: "Input dimensions are incompatible for singleton expansion.",
    message: "bsxfun: input sizes are not compatible for singleton expansion",
};
pub const BSXFUN_ERROR_FUNCTION_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BSXFUN.FUNCTION_ERROR",
    identifier: Some("RunMat:bsxfun:FunctionError"),
    when: "The callback fails or returns non-scalar or non-uniform values.",
    message: "bsxfun: callback execution error",
};
pub const BSXFUN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BSXFUN.INTERNAL",
    identifier: Some("RunMat:bsxfun:Internal"),
    when: "Output allocation or provider gathering fails internally.",
    message: "bsxfun: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    BSXFUN_ERROR_INVALID_FUNCTION,
    BSXFUN_ERROR_INVALID_INPUT,
    BSXFUN_ERROR_SIZE_MISMATCH,
    BSXFUN_ERROR_FUNCTION_ERROR,
    BSXFUN_ERROR_INTERNAL,
];

pub const BSXFUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const BSXFUN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [BuiltinExtensionDescriptor {
    id: "bsxfun-text-callable",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bsxfun with a text callback name is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BsxfunTextCallableExtension"),
}];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "A", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger, notes: "The selected callback owns mixed-class admission, arithmetic, overflow, and output class." },
    BuiltinIntegerInputCapability { name: "B", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger, notes: "Singleton expansion passes an exact scalar of the stored class to each callback invocation." },
];
pub const BSXFUN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "C = bsxfun(fun,A,B) with integer A or B",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer values remain exact through scalar extraction and uniform result collection; the callback's canonical contract determines the result.",
    }];
