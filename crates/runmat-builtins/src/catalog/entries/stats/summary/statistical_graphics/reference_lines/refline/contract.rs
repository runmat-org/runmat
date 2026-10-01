//! Reference-line forms, named extensions, and exact graphics-boundary inputs.
use crate::*;

const HANDLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Line graphics handle or vector of handles.",
};
const AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
};
const COEFFS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "coeffs",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element vector [slope intercept].",
};
const SLOPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "m",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Line slope.",
};
const INTERCEPT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Line y-intercept.",
};
const STYLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "style",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional LineSpec or line Name/Value pairs.",
};
const EMPTY: [BuiltinParamDescriptor; 0] = [];
const AX_INPUT: [BuiltinParamDescriptor; 1] = [AX];
const COEFFS_INPUT: [BuiltinParamDescriptor; 1] = [COEFFS];
const SLOPE_INTERCEPT_INPUT: [BuiltinParamDescriptor; 2] = [SLOPE, INTERCEPT];
const COEFFS_STYLE_INPUT: [BuiltinParamDescriptor; 2] = [COEFFS, STYLE];
const SLOPE_INTERCEPT_STYLE_INPUT: [BuiltinParamDescriptor; 3] = [SLOPE, INTERCEPT, STYLE];
const AX_STYLE_INPUT: [BuiltinParamDescriptor; 2] = [AX, STYLE];
const OUTPUT: [BuiltinParamDescriptor; 1] = [HANDLE];
const SIGNATURES: [BuiltinSignatureDescriptor; 11] = [
    BuiltinSignatureDescriptor {
        label: "refline()",
        inputs: &EMPTY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "refline(ax)",
        inputs: &AX_INPUT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "refline(coeffs)",
        inputs: &COEFFS_INPUT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "refline(m, b)",
        inputs: &SLOPE_INTERCEPT_INPUT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "refline(___, LineSpec, Name, Value)",
        inputs: &COEFFS_STYLE_INPUT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "refline(ax, ___)",
        inputs: &AX_STYLE_INPUT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = refline()",
        inputs: &EMPTY,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = refline(coeffs)",
        inputs: &COEFFS_INPUT,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = refline(m, b)",
        inputs: &SLOPE_INTERCEPT_INPUT,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = refline(m, b, LineSpec)",
        inputs: &SLOPE_INTERCEPT_STYLE_INPUT,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = refline(ax, ___)",
        inputs: &AX_STYLE_INPUT,
        outputs: &OUTPUT,
    },
];
pub const REFLINE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REFLINE.INVALID_ARGUMENT",
    identifier: Some("RunMat:refline:InvalidArgument"),
    when: "Axes handle, coefficient inputs, or style options are malformed.",
    message: "refline: invalid argument",
};
pub const REFLINE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REFLINE.INTERNAL",
    identifier: Some("RunMat:refline:Internal"),
    when: "RunMat cannot construct or register the reference line.",
    message: "refline: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 2] =
    [REFLINE_ERROR_INVALID_ARGUMENT, REFLINE_ERROR_INTERNAL];
pub const REFLINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const REFLINE_INTEGER_COEFFICIENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "refline-integer-coefficients",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "refline accepts typed-integer slope/intercept coefficients as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ReflineIntegerCoefficientsExtension"),
    };
pub const REFLINE_EXPLICIT_GPU_COEFFICIENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "refline-explicit-gpu-coefficients",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "refline accepts explicitly GPU-resident coefficients as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ReflineExplicitGpuCoefficientsExtension"),
    };
pub const REFLINE_LOGICAL_COEFFICIENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "refline-logical-coefficients",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "refline accepts logical slope/intercept coefficients as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ReflineLogicalCoefficientsExtension"),
    };
pub const REFLINE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    REFLINE_INTEGER_COEFFICIENT_EXTENSION,
    REFLINE_EXPLICIT_GPU_COEFFICIENT_EXTENSION,
    REFLINE_LOGICAL_COEFFICIENT_EXTENSION,
];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "m", classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents single and double slopes. RunMat admits typed integers only at a checked binary64 graphics boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "b/coeffs", classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The intercept or coefficient vector follows the same independently gated checked floating boundary.",
    },
];
pub const REFLINE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "h = refline(integer_coeffs) or refline(integer_m,integer_b)",
    inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
    output_class: BuiltinIntegerOutputClassRule::NotApplicable,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "Slope/intercept arithmetic and line geometry are explicit floating boundaries. Automatic residency gathers transparently; explicit GPU coefficients are independently gated.",
}];
