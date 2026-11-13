//! Canonical peaks signatures, errors, extensions, and integer forms.
use crate::*;

const PEAKS_OUTPUT_Z: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sampled peaks surface values.",
}];

const PEAKS_OUTPUT_XY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid coordinates along X-axis.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid coordinates along Y-axis.",
    },
];

const PEAKS_OUTPUT_XYZ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid coordinates along X-axis.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid coordinates along Y-axis.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sampled peaks surface values.",
    },
];

const PEAKS_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];

const PEAKS_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: Some("49"),
    description: "Grid size n for n-by-n sampling.",
}];

const PEAKS_SIG_XY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate matrix.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate matrix (same size as X).",
    },
];

const PEAKS_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "Z = peaks()",
        inputs: &PEAKS_SIG_EMPTY_INPUTS,
        outputs: &PEAKS_OUTPUT_Z,
    },
    BuiltinSignatureDescriptor {
        label: "Z = peaks(n)",
        inputs: &PEAKS_SIG_N_INPUTS,
        outputs: &PEAKS_OUTPUT_Z,
    },
    BuiltinSignatureDescriptor {
        label: "Z = peaks(X, Y)",
        inputs: &PEAKS_SIG_XY_INPUTS,
        outputs: &PEAKS_OUTPUT_Z,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y] = peaks(n)",
        inputs: &PEAKS_SIG_N_INPUTS,
        outputs: &PEAKS_OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y,Z] = peaks(n)",
        inputs: &PEAKS_SIG_N_INPUTS,
        outputs: &PEAKS_OUTPUT_XYZ,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y] = peaks(X, Y)",
        inputs: &PEAKS_SIG_XY_INPUTS,
        outputs: &PEAKS_OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y,Z] = peaks(X, Y)",
        inputs: &PEAKS_SIG_XY_INPUTS,
        outputs: &PEAKS_OUTPUT_XYZ,
    },
];

const PEAKS_ERRORS: [BuiltinErrorDescriptor; 10] = [
    BuiltinErrorDescriptor {
        code: "RM.PEAKS.OUTPUT_COUNT",
        identifier: None,
        when: "More than three outputs are requested.",
        message: "peaks: too many output arguments; maximum is 3",
    },
    BuiltinErrorDescriptor {
        code: "RM.PEAKS.ARG_COUNT",
        identifier: None,
        when: "The builtin is called with more than two input arguments.",
        message: "peaks: expected 0, 1, or 2 input arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.PEAKS.N_NON_SCALAR",
        identifier: None,
        when: "The n argument is not a numeric scalar.",
        message: "peaks: n must be a numeric scalar",
    },
    BuiltinErrorDescriptor {
        code: "RM.PEAKS.N_NON_FINITE",
        identifier: None,
        when: "The n argument is non-finite.",
        message: "peaks: n must be finite",
    },
    BuiltinErrorDescriptor {
        code: "RM.PEAKS.N_NON_INTEGER",
        identifier: None,
        when: "The n argument is not an integer.",
        message: "peaks: n must be an integer",
    },
    BuiltinErrorDescriptor {
        code: "RM.PEAKS.N_OUT_OF_RANGE",
        identifier: None,
        when: "The scalar n argument is not greater than one.",
        message: "peaks: scalar n must be an integer greater than 1",
    },
    BuiltinErrorDescriptor {
        code: "RM.PEAKS.N_TOO_LARGE",
        identifier: None,
        when: "The n argument exceeds platform limits.",
        message: "peaks: n is too large for this platform",
    },
    BuiltinErrorDescriptor {
        code: "RM.PEAKS.XY_NON_NUMERIC",
        identifier: None,
        when: "X/Y inputs are not numeric matrix values.",
        message: "peaks: X and Y must be numeric matrices",
    },
    BuiltinErrorDescriptor {
        code: "RM.PEAKS.XY_NOT_2D",
        identifier: None,
        when: "X/Y inputs are not 2-D matrices.",
        message: "peaks: X and Y must be 2-D matrices",
    },
    BuiltinErrorDescriptor {
        code: "RM.PEAKS.XY_SIZE_MISMATCH",
        identifier: None,
        when: "X and Y inputs do not have the same shape.",
        message: "peaks: X and Y must have the same size",
    },
];

pub const PEAKS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PEAKS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PEAKS_ERRORS,
};

pub const PEAKS_TYPED_INTEGER_N_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "peaks-typed-integer-grid-size",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "peaks accepts a typed-integer scalar grid size as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PeaksTypedIntegerGridSizeExtension"),
    };

pub const PEAKS_EXPLICIT_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "peaks-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "peaks accepts explicit gpuArray coordinate or grid-size inputs as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PeaksExplicitGpuInputExtension"),
    };

pub const PEAKS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    PEAKS_TYPED_INTEGER_N_EXTENSION,
    PEAKS_EXPLICIT_GPU_INPUT_EXTENSION,
];

const PEAKS_INTEGER_N_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public scalar form requires an integer value greater than one but does not enumerate typed integer storage. RunMat decodes all eight classes exactly as a structural grid size.",
    }];

const PEAKS_INTEGER_XY_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "Xm",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target explicitly documents all eight integer coordinate classes. X output preserves Xm exactly when requested, while Z crosses the peaks floating formula boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "Ym",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target explicitly documents all eight integer coordinate classes. Y output preserves Ym exactly when requested, while Z crosses the peaks floating formula boundary.",
    },
];

pub const PEAKS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[X,Y,Z] = peaks(integer_n)",
        inputs: &PEAKS_INTEGER_N_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Typed n is a gated structural extension; generated coordinates and surface values are floating and provider selection remains automatic.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[X,Y,Z] = peaks(integer_Xm, integer_Ym)",
        inputs: &PEAKS_INTEGER_XY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Documented integer coordinates remain authoritative for returned X and Y; Z intentionally materializes a floating view for transcendental evaluation. Explicit gpuArray inputs are separately RunMat-only because the public page does not advertise GPU arrays.",
    },
];
