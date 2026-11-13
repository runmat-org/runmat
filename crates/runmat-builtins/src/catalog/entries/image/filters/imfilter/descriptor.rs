use crate::*;

const IMFILTER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filtered image output.",
}];

const IMFILTER_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input image/array.",
    },
    BuiltinParamDescriptor {
        name: "H",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter kernel.",
    },
];

const IMFILTER_INPUTS_WITH_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input image/array.",
    },
    BuiltinParamDescriptor {
        name: "H",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter kernel.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Optional mode/shape/padding flags and fill values (e.g. 'same', 'full', 'valid', 'replicate', 'symmetric', 'circular', 'fill', padval, 'conv', 'corr').",
    },
];

const IMFILTER_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = imfilter(A, H)",
        inputs: &IMFILTER_INPUTS_CORE,
        outputs: &IMFILTER_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = imfilter(A, H, options...)",
        inputs: &IMFILTER_INPUTS_WITH_OPTIONS,
        outputs: &IMFILTER_OUTPUT,
    },
];

pub const IMFILTER_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMFILTER.INVALID_INPUT",
    identifier: Some("RunMat:imfilter:InvalidInput"),
    when: "Input/kernel tensors are invalid or cannot be converted.",
    message: "imfilter: invalid input",
};

pub const IMFILTER_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMFILTER.INVALID_OPTION",
    identifier: Some("RunMat:imfilter:InvalidOption"),
    when: "One or more option flags or option values are invalid.",
    message: "imfilter: invalid option",
};

pub const IMFILTER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMFILTER.INTERNAL",
    identifier: Some("RunMat:imfilter:Internal"),
    when: "Internal filtering operation fails.",
    message: "imfilter: internal operation failed",
};

const IMFILTER_ERRORS: [BuiltinErrorDescriptor; 3] = [
    IMFILTER_ERROR_INVALID_INPUT,
    IMFILTER_ERROR_INVALID_OPTION,
    IMFILTER_ERROR_INTERNAL,
];

pub const IMFILTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMFILTER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IMFILTER_ERRORS,
};

pub const IMFILTER_VALID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "imfilter-valid-output-shape",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "imfilter(..., 'valid') is a RunMat-only output-shape extension",
    error_identifier: Some("RunMat:compatibility:ImfilterValidExtension"),
};

pub const IMFILTER_FILL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "imfilter-fill-keyword",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "imfilter(..., 'fill', padval) is a RunMat-only spelling; MATLAB accepts padval directly",
    error_identifier: Some("RunMat:compatibility:ImfilterFillExtension"),
};

pub const IMFILTER_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [IMFILTER_VALID_EXTENSION, IMFILTER_FILL_EXTENSION];

const IMFILTER_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented int8, int16, int32, uint8, uint16, and uint32 images retain their authoritative class. Host and automatic-fallback calls accumulate in binary64; explicit gpuArray calls accumulate in binary64 for uint32 and binary32 for the other documented integer classes.",
    }];

pub const IMFILTER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "B = imfilter(integer_A, double_h, options...)" ,
        inputs: &IMFILTER_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host and automatic-placement fallback sums use binary64 so planner placement is transparent. Explicit gpuArray uint32 sums use binary64, while the other documented explicit resident integer classes use binary32 multiplication and accumulation. Fractional results round to nearest and out-of-range results saturate before authoritative same-class storage is constructed; owner-aware fallback preserves explicit residency.",
    }];
