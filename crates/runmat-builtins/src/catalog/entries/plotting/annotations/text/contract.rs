use crate::*;
pub const INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "text-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "text with a typed-integer numeric axes alias is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TextIntegerAxesHandleExtension"),
};
pub const TEXT_EXPLICIT_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "text-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "text with explicit gpuArray coordinates is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TextExplicitGpuInputExtension"),
    };
pub const TEXT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [INTEGER_AXES_EXTENSION, TEXT_EXPLICIT_GPU_INPUT_EXTENSION];

const INTEGER_COORDINATE_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public Position property explicitly accepts all eight integer classes; native coordinate identity is retained before the renderer boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer y coordinates are documented graphics data and cross to f32 only in the derived render cache.",
    },
    BuiltinIntegerInputCapability {
        name: "z",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer z coordinates are documented graphics data and cross to f32 only in the derived render cache.",
    },
];
const INTEGER_AXES_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "ax",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "MATLAB axes are graphics objects; typed-integer numeric aliases are a gated RunMat representation extension.",
}];
pub const TEXT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = text(integer_x, integer_y [, integer_z], label, ...)",
        inputs: &INTEGER_COORDINATE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Documented native coordinates remain the semantic source while the plotting backend receives an explicitly derived f32 position.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = text(integer_ax, x, y [, z], label, ...)",
        inputs: &INTEGER_AXES_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The axes-alias extension gate runs before target selection or graphics mutation.",
    },
];

const TEXT_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created text annotation.",
}];

const TEXT_INPUTS_X_Y_LABEL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate.",
    },
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Annotation text.",
    },
];

const TEXT_INPUTS_X_Y_Z_LABEL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate.",
    },
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinate.",
    },
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Annotation text.",
    },
];

const TEXT_INPUTS_X_Y_LABEL_PROPS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate.",
    },
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Annotation text.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value style options.",
    },
];

const TEXT_INPUTS_X_Y_Z_LABEL_PROPS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate.",
    },
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinate.",
    },
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Annotation text.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value style options.",
    },
];

const TEXT_INPUTS_AX_X_Y_LABEL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate.",
    },
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Annotation text.",
    },
];

const TEXT_INPUTS_AX_X_Y_Z_LABEL: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinate.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinate.",
    },
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinate.",
    },
    BuiltinParamDescriptor {
        name: "label",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Annotation text.",
    },
];

const TEXT_SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "h = text(x, y, label)",
        inputs: &TEXT_INPUTS_X_Y_LABEL,
        outputs: &TEXT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = text(x, y, label, Name, Value, ...)",
        inputs: &TEXT_INPUTS_X_Y_LABEL_PROPS,
        outputs: &TEXT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = text(x, y, z, label)",
        inputs: &TEXT_INPUTS_X_Y_Z_LABEL,
        outputs: &TEXT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = text(x, y, z, label, Name, Value, ...)",
        inputs: &TEXT_INPUTS_X_Y_Z_LABEL_PROPS,
        outputs: &TEXT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = text(ax, x, y, label)",
        inputs: &TEXT_INPUTS_AX_X_Y_LABEL,
        outputs: &TEXT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = text(ax, x, y, z, label)",
        inputs: &TEXT_INPUTS_AX_X_Y_Z_LABEL,
        outputs: &TEXT_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "text(x, y, label)",
        inputs: &TEXT_INPUTS_X_Y_LABEL,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "text(x, y, label, Name, Value, ...)",
        inputs: &TEXT_INPUTS_X_Y_LABEL_PROPS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "text(x, y, z, label)",
        inputs: &TEXT_INPUTS_X_Y_Z_LABEL,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "text(x, y, z, label, Name, Value, ...)",
        inputs: &TEXT_INPUTS_X_Y_Z_LABEL_PROPS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "text(ax, x, y, label)",
        inputs: &TEXT_INPUTS_AX_X_Y_LABEL,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "text(ax, x, y, z, label)",
        inputs: &TEXT_INPUTS_AX_X_Y_Z_LABEL,
        outputs: &[],
    },
];

pub const TEXT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT.INVALID_ARGUMENT",
    identifier: Some("RunMat:text:InvalidArgument"),
    when: "Coordinate, label, axes-target, or name/value text style arguments are invalid.",
    message: "text: invalid argument",
};

pub const TEXT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT.INTERNAL",
    identifier: Some("RunMat:text:Internal"),
    when: "Internal plotting state update fails while creating annotation.",
    message: "text: internal operation failed",
};

const TEXT_ERRORS: [BuiltinErrorDescriptor; 3] = [
    TEXT_ERROR_INVALID_ARGUMENT,
    TEXT_ERROR_INTERNAL,
    TEXT_ERROR_TOO_MANY_OUTPUTS,
];
pub const TEXT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:text:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "text: too many output arguments",
};

pub const TEXT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TEXT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TEXT_ERRORS,
};
