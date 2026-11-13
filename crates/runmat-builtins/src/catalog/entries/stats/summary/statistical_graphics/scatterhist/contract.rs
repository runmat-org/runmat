//! Scatterhist public forms, diagnostics, compatibility and integer contracts.
use crate::*;

pub const SCATTERHIST_MAX_BINS: usize = 250;

const OUTPUT_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Three axes handles: scatter, x marginal, and y marginal.",
};

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "X observations.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Y observations.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Scatterhist name-value options.",
};

const INPUT_XY: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUT_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_OPTIONS];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [OUTPUT_AX];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "scatterhist(x, y)",
        inputs: &INPUT_XY,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "scatterhist(x, y, Name, Value)",
        inputs: &INPUT_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = scatterhist(___)",
        inputs: &INPUT_OPTIONS,
        outputs: &OUTPUTS,
    },
];

pub const SCATTERHIST_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SCATTERHIST.INVALID_ARGUMENT",
    identifier: Some("RunMat:scatterhist:InvalidArgument"),
    when: "Observation vectors or name-value options are malformed.",
    message: "scatterhist: invalid argument",
};

pub const SCATTERHIST_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SCATTERHIST.INTERNAL",
    identifier: Some("RunMat:scatterhist:Internal"),
    when: "RunMat cannot allocate or render the scatterhist chart.",
    message: "scatterhist: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [
    SCATTERHIST_ERROR_INVALID_ARGUMENT,
    SCATTERHIST_ERROR_INTERNAL,
];

pub const SCATTERHIST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const SCATTERHIST_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "scatterhist-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "scatterhist accepts typed-integer observation data as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ScatterhistIntegerDataExtension"),
    };
pub const SCATTERHIST_INTEGER_GROUP_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "scatterhist-integer-group",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "scatterhist accepts typed-integer Group labels as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ScatterhistIntegerGroupExtension"),
    };
pub const SCATTERHIST_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "scatterhist-integer-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "scatterhist accepts typed-integer numeric option values as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ScatterhistIntegerOptionExtension"),
    };
pub const SCATTERHIST_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "scatterhist-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "scatterhist accepts interactive gpuArray arguments as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ScatterhistGpuInputExtension"),
    };
pub const SCATTERHIST_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    SCATTERHIST_INTEGER_DATA_EXTENSION,
    SCATTERHIST_INTEGER_GROUP_EXTENSION,
    SCATTERHIST_INTEGER_OPTION_EXTENSION,
    SCATTERHIST_GPU_INPUT_EXTENSION,
];
const SCATTERHIST_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "x",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double observations; typed integers require exact binary64 representation before histogram and rendering computation.",
    },
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The same checked boundary applies independently to Y observations.",
    },
];
const SCATTERHIST_INTEGER_GROUP_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Group",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer group identity is retained exactly, including wide int64 and uint64 values, and converted only to textual legend labels.",
    }];
const SCATTERHIST_INTEGER_OPTION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "NBins or numeric style/control value",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents these numeric controls as single or double. Exact structural flags/bin counts remain integer until validation; renderer-domain sizes, widths, bandwidths, and RGB values require checked binary64 conversion.",
    }];
pub const SCATTERHIST_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = scatterhist(integer_x, integer_y, ...)",
        inputs: &SCATTERHIST_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The RunMat-only observation extension crosses once into double histogram/statistical geometry after exactness checks; the public result is a vector of double graphics handles.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = scatterhist(x,y,\"Group\",integer_group)",
        inputs: &SCATTERHIST_INTEGER_GROUP_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Grouping compares exact native values and therefore does not collapse distinct wide integers through double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = scatterhist(x,y,option_name,integer_value,...)",
        inputs: &SCATTERHIST_INTEGER_OPTION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Each RunMat-only typed option is gated before gather; exact structural controls and checked renderer-domain conversions are distinguished by option role.",
    },
];
