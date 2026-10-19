use crate::*;

pub const SUBPLOT_INTEGER_GRID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "subplot-integer-grid-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow typed-integer subplot grid controls",
    error_identifier: Some("RunMat:compatibility:SubplotIntegerGridControlExtension"),
};
pub const SUBPLOT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [SUBPLOT_INTEGER_GRID_EXTENSION];
const SUBPLOT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "rows, cols, or position",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target's public data types for these controls are single and double; RunMat mode additionally parses typed integer scalars exactly.",
    }];
pub const SUBPLOT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "ax = subplot(integer_rows, integer_cols, integer_position)",
        inputs: &SUBPLOT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The gated scalars are converted directly to platform dimensions before any figure state changes; the graphics handle remains a host double scalar.",
    }];

const SUBPLOT_OUTPUT_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Encoded handle for the selected subplot axes.",
}];

const SUBPLOT_INPUTS_GRID: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "rows",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of subplot rows.",
    },
    BuiltinParamDescriptor {
        name: "cols",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of subplot columns.",
    },
    BuiltinParamDescriptor {
        name: "position",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "1-based subplot position in row-major order.",
    },
];

const SUBPLOT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ax = subplot(rows, cols, position)",
    inputs: &SUBPLOT_INPUTS_GRID,
    outputs: &SUBPLOT_OUTPUT_AX,
}];

pub const SUBPLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:subplot:InvalidArgument"),
    when: "Rows/cols/position are unsupported, non-scalar, non-finite, or non-positive.",
    message: "subplot: invalid argument",
};

pub const SUBPLOT_ERROR_INDEX_OUT_OF_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBPLOT.INDEX_OUT_OF_RANGE",
    identifier: Some("RunMat:subplot:IndexOutOfRange"),
    when: "Position is outside the configured subplot grid.",
    message: "subplot: subplot index is out of range for the grid",
};

pub const SUBPLOT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBPLOT.INTERNAL",
    identifier: Some("RunMat:subplot:Internal"),
    when: "Internal subplot state operation fails.",
    message: "subplot: internal operation failed",
};

const SUBPLOT_ERRORS: [BuiltinErrorDescriptor; 4] = [
    SUBPLOT_ERROR_INVALID_ARGUMENT,
    SUBPLOT_ERROR_INDEX_OUT_OF_RANGE,
    SUBPLOT_ERROR_INTERNAL,
    SUBPLOT_ERROR_TOO_MANY_OUTPUTS,
];

pub const SUBPLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBPLOT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUBPLOT_ERRORS,
};
pub const SUBPLOT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUBPLOT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:subplot:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "subplot: at most one output is supported",
};
