//! Typed declarations conserved from the Runtime owner.
use crate::*;

pub const TRND_INTEGER_NU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "trnd-integer-degrees-of-freedom",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "trnd with typed-integer degrees of freedom is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TrndIntegerDegreesOfFreedomExtension"),
};

pub const TRND_INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "trnd-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "trnd with typed-integer size arguments is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TrndIntegerSizeExtension"),
};

pub const TRND_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [TRND_INTEGER_NU_EXTENSION, TRND_INTEGER_SIZE_EXTENSION];

pub const OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random sample array from the Student's t distribution.",
}];

pub const INPUT_NU: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nu",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Degrees of freedom parameter.",
};

pub const INPUT_SZ: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Output size arguments.",
};

pub const INPUTS_NU: [BuiltinParamDescriptor; 1] = [INPUT_NU];

pub const INPUTS_NU_SZ: [BuiltinParamDescriptor; 2] = [INPUT_NU, INPUT_SZ];

pub const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = trnd(nu)",
        inputs: &INPUTS_NU,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = trnd(nu, sz)",
        inputs: &INPUTS_NU_SZ,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = trnd(nu, sz1, sz2, ...)",
        inputs: &INPUTS_NU_SZ,
        outputs: &OUTPUT_R,
    },
];

pub const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRND.INVALID_ARGUMENT",
    identifier: Some("RunMat:trnd:InvalidArgument"),
    when: "Input parameters or size arguments are missing, malformed, or incompatible.",
    message: "trnd: invalid argument",
};

pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRND.INTERNAL",
    identifier: Some("RunMat:trnd:Internal"),
    when: "Internal tensor conversion or allocation fails.",
    message: "trnd: internal operation failed",
};

pub const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const TRND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "nu",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer degrees of freedom are gated by trnd-integer-degrees-of-freedom and enter the floating Student-t computation domain.",
    },
    BuiltinIntegerInputCapability {
        name: "sz",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer scalar/vector size arguments are gated by trnd-integer-size and parsed exactly before allocation.",
    },
];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "r = trnd(nu, sz1, sz2, ...)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed-integer parameters are RunMat-only; integer nu produces double samples, and a resident source receives host-generated fallback output re-uploaded with matching precision.",
    }];
