use crate::*;

const COLORORDER_OUTPUT_COLORS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current color order as an m-by-3 RGB triplet matrix.",
}];
const COLORORDER_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const COLORORDER_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const COLORORDER_INPUTS_COLORS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "colors",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "m-by-3 RGB matrix, color name array, color cellstr, palette name, or default.",
}];

const COLORORDER_INPUTS_TARGET: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "target",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target figure or axes handle.",
}];

const COLORORDER_INPUTS_TARGET_COLORS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "target",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target figure or axes handle.",
    },
    BuiltinParamDescriptor {
        name: "colors",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "m-by-3 RGB matrix, color name array, color cellstr, palette name, or default.",
    },
];

const COLORORDER_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "C = colororder()",
        inputs: &COLORORDER_INPUTS_NONE,
        outputs: &COLORORDER_OUTPUT_COLORS,
    },
    BuiltinSignatureDescriptor {
        label: "C = colororder(target)",
        inputs: &COLORORDER_INPUTS_TARGET,
        outputs: &COLORORDER_OUTPUT_COLORS,
    },
    BuiltinSignatureDescriptor {
        label: "colororder(colors)",
        inputs: &COLORORDER_INPUTS_COLORS,
        outputs: &COLORORDER_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "colororder(palettename)",
        inputs: &COLORORDER_INPUTS_COLORS,
        outputs: &COLORORDER_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "colororder(target, colors)",
        inputs: &COLORORDER_INPUTS_TARGET_COLORS,
        outputs: &COLORORDER_OUTPUTS_NONE,
    },
];

const COLORORDER_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORORDER.INVALID_ARGUMENT",
    identifier: Some("RunMat:colororder:InvalidArgument"),
    when: "Target handle, color array, palette name, or color specification is invalid.",
    message: "colororder: invalid argument",
};

const COLORORDER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORORDER.INTERNAL",
    identifier: Some("RunMat:colororder:Internal"),
    when: "Internal plotting state update fails.",
    message: "colororder: internal operation failed",
};

const COLORORDER_ERRORS: [BuiltinErrorDescriptor; 3] = [
    COLORORDER_ERROR_INVALID_ARGUMENT,
    COLORORDER_ERROR_INTERNAL,
    COLORORDER_ERROR_TOO_MANY_OUTPUTS,
];

pub const COLORORDER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COLORORDER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COLORORDER_ERRORS,
};

pub const COLORORDER_INTEGER_RGB_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "colororder-integer-rgb",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "colororder with a typed integer RGB matrix is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ColororderIntegerRgbExtension"),
    };

pub const COLORORDER_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    COLORORDER_INTEGER_RGB_EXTENSION,
    AXES_CONTROL_SETTER_OUTPUT_EXTENSION,
];

pub const COLORORDER_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "colors",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public numeric RGB matrices are single or double. RunMat mode additionally admits every integer class when each channel is exactly 0 or 1.",
    }];

pub const COLORORDER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "colororder(integer_colors)",
        inputs: &COLORORDER_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Integer channels are range-checked from native storage, then cross the explicit host graphics-state f32 boundary; subsequent query output is an m-by-3 host double matrix. Resident color arrays reject without provider access.",
    }];
pub const COLORORDER_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORORDER.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:colororder:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "colororder: at most one output is supported",
};
