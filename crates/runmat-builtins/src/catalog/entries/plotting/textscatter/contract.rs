use crate::*;

pub const TEXTSCATTER_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "textscatter-integer-axes-handle",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "textscatter with a typed-integer numeric axes alias is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TextScatterIntegerAxesHandleExtension"),
    };

pub const TEXTSCATTER_EXPLICIT_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "textscatter-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "textscatter with explicit gpuArray input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TextScatterExplicitGpuInputExtension"),
    };

pub const TEXTSCATTER_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    TEXTSCATTER_INTEGER_AXES_EXTENSION,
    TEXTSCATTER_EXPLICIT_GPU_INPUT_EXTENSION,
];

const INTEGER_COORDINATE_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "XData",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented coordinate data and remain authoritative outside the derived render cache.",
    },
    BuiltinIntegerInputCapability {
        name: "YData",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented coordinate data and remain authoritative outside the derived render cache.",
    },
    BuiltinIntegerInputCapability {
        name: "ZData",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "ZData applies to textscatter3 and retains its native integer class and exact values.",
    },
];

const INTEGER_AXES_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "ax",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "MATLAB axes are graphics objects; typed-integer numeric aliases are a gated RunMat representation extension.",
}];

pub const TEXTSCATTER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "ts = textscatter(integer_coordinates, labels, ...)",
        inputs: &INTEGER_COORDINATE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented coordinate properties retain native source storage; only annotation and marker geometry enter the floating renderer domain.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ts = textscatter(integer_ax, ...)",
        inputs: &INTEGER_AXES_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The axes-alias extension gate runs before target selection or graphics mutation.",
    },
];

const OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ts",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the TextScatter chart object.",
}];

const INPUTS_TEXTSCATTER_XY: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String vector or cell array of character vectors.",
    },
];

const INPUTS_TEXTSCATTER_MATRIX: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "xy",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "N-by-2 coordinate matrix.",
    },
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String vector or cell array of character vectors.",
    },
];

const INPUTS_PROPS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "props",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name/value TextScatter properties.",
}];

const TEXTSCATTER_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "ts = textscatter(x, y, str)",
        inputs: &INPUTS_TEXTSCATTER_XY,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ts = textscatter(xy, str)",
        inputs: &INPUTS_TEXTSCATTER_MATRIX,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ts = textscatter(ax, ...)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ts = textscatter(..., Name, Value)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT_HANDLE,
    },
];

pub const TEXTSCATTER_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCATTER.INVALID_ARGUMENT",
    identifier: Some("RunMat:textscatter:InvalidArgument"),
    when: "Coordinates, text labels, axes target, or TextScatter property values are invalid.",
    message: "textscatter: invalid argument",
};

pub const TEXTSCATTER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCATTER.INTERNAL",
    identifier: Some("RunMat:textscatter:Internal"),
    when: "Internal plotting state update fails while creating or mutating TextScatter.",
    message: "textscatter: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    TEXTSCATTER_ERROR_INVALID_ARGUMENT,
    TEXTSCATTER_ERROR_INTERNAL,
    TEXTSCATTER_ERROR_TOO_MANY_OUTPUTS,
];

pub const TEXTSCATTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TEXTSCATTER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const TEXTSCATTER_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCATTER.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:textscatter:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "textscatter: too many output arguments",
};
