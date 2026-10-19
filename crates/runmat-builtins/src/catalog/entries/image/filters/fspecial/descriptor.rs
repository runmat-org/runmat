use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "H",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Generated 2-D correlation kernel.",
}];
const KIND: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "type",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Filter name: average, disk, gaussian, laplacian, log, motion, prewitt, sobel, or unsharp.",
}];
const KIND_ARG1: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor { name: "type", ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required, default: None,
        description: "Filter name: average, disk, gaussian, laplacian, log, motion, prewitt, sobel, or unsharp." },
    BuiltinParamDescriptor { name: "arg1", ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional, default: None,
        description: "Filter-specific size, radius, alpha, or length." },
];
const KIND_ARG2: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor { name: "type", ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required, default: None,
        description: "Filter name: average, disk, gaussian, laplacian, log, motion, prewitt, sobel, or unsharp." },
    BuiltinParamDescriptor { name: "arg1", ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional, default: None,
        description: "First filter-specific parameter." },
    BuiltinParamDescriptor { name: "arg2", ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional, default: None,
        description: "Second filter-specific parameter, such as sigma or motion angle." },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "H = fspecial(type)",
        inputs: &KIND,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "H = fspecial(type, arg1)",
        inputs: &KIND_ARG1,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "H = fspecial(type, arg1, arg2)",
        inputs: &KIND_ARG2,
        outputs: &OUTPUT,
    },
];
pub const FSPECIAL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSPECIAL.INVALID_ARGUMENT",
    identifier: Some("RunMat:fspecial:InvalidArgument"),
    when: "Filter name or argument count shape is invalid for the selected filter type.",
    message: "fspecial: invalid argument",
};
pub const FSPECIAL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSPECIAL.INVALID_INPUT",
    identifier: Some("RunMat:fspecial:InvalidInput"),
    when: "Filter parameters have invalid values or unsupported input types.",
    message: "fspecial: invalid input",
};
pub const FSPECIAL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSPECIAL.INTERNAL",
    identifier: Some("RunMat:fspecial:Internal"),
    when: "Kernel generation or provider publication fails internally.",
    message: "fspecial: internal kernel generation failure",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    FSPECIAL_ERROR_INVALID_ARGUMENT,
    FSPECIAL_ERROR_INVALID_INPUT,
    FSPECIAL_ERROR_INTERNAL,
];
pub const FSPECIAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FSPECIAL_NONDOUBLE_SIZE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fspecial-nondouble-size",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "fspecial with a single, integer, or logical size control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FspecialNondoubleSizeExtension"),
    };
pub const FSPECIAL_NONDOUBLE_PARAMETER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fspecial-nondouble-parameter", mode: BuiltinExtensionMode::RunMatOnly,
    description: "fspecial with a single, integer, or logical computational parameter is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FspecialNondoubleParameterExtension"),
};
pub const FSPECIAL_UNSHARP_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fspecial-unsharp-filter",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fspecial with the unsharp filter type is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FspecialUnsharpExtension"),
};
pub const FSPECIAL_RESIDENT_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fspecial-resident-output", mode: BuiltinExtensionMode::RunMatOnly,
    description: "fspecial evaluated under ambient interactive resident-output policy is a RunMat extension; unsupported provider forms fall back to host output",
    error_identifier: Some("RunMat:compatibility:FspecialResidentOutputExtension"),
};
pub const FSPECIAL_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FSPECIAL_NONDOUBLE_SIZE_EXTENSION,
    FSPECIAL_NONDOUBLE_PARAMETER_EXTENSION,
    FSPECIAL_UNSHARP_EXTENSION,
    FSPECIAL_RESIDENT_OUTPUT_EXTENSION,
];

const SIZE_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "hsize or len", classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "All eight integer classes are decoded from authoritative storage as exact positive structural dimensions.",
}];
const PARAMETER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "radius, sigma, alpha, or theta", classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "All eight integer classes are admitted only when exactly representable at the binary64 kernel-computation boundary.",
}];
pub const FSPECIAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "H = fspecial(type, integer_hsize_or_len, ...)", inputs: &SIZE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only exact structural controls; host kernels are double, while the independently gated resident-output extension follows provider F32/F64 precision when supported.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "H = fspecial(type, integer_parameter, ...)", inputs: &PARAMETER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "RunMat-only computational controls cross one exact binary64 boundary; host kernels are double, while the independently gated resident-output extension follows provider F32/F64 precision when supported.",
    },
];
