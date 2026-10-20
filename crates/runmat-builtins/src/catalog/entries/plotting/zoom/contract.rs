use crate::*;

pub const ZOOM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ZOOM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ZOOM_ERRORS,
};

pub const ZOOM_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "z = zoom()",
        inputs: &ZOOM_INPUTS_NONE,
        outputs: &ZOOM_OUTPUT_OBJECT,
    },
    BuiltinSignatureDescriptor {
        label: "zoom()",
        inputs: &ZOOM_INPUTS_NONE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = zoom(option)",
        inputs: &ZOOM_INPUTS_OPTION,
        outputs: &ZOOM_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = zoom(factor)",
        inputs: &ZOOM_INPUTS_FACTOR,
        outputs: &ZOOM_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = zoom(target, option)",
        inputs: &ZOOM_INPUTS_TARGET_OPTION,
        outputs: &ZOOM_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "z = zoom(fig)",
        inputs: &ZOOM_INPUTS_FIGURE,
        outputs: &ZOOM_OUTPUT_OBJECT,
    },
];

pub const ZOOM_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

pub const ZOOM_OUTPUT_OBJECT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "z",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zoom mode object for the current or specified figure.",
}];

pub const ZOOM_INPUTS_OPTION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zoom option: on, off, toggle, xon, yon, reset, or out.",
}];

pub const ZOOM_OUTPUT_STATUS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Returns \"ok\" when the zoom command succeeds.",
}];

pub const ZOOM_INPUTS_FACTOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "factor",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive numeric zoom factor.",
}];

pub const ZOOM_INPUTS_TARGET_OPTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "target",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Figure or axes handle.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zoom option or factor.",
    },
];

pub const ZOOM_INPUTS_FIGURE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fig",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target figure handle.",
}];

pub const ZOOM_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ZOOM_ERROR_INVALID_ARGUMENT, ZOOM_ERROR_INTERNAL];

pub const ZOOM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZOOM.INVALID_ARGUMENT",
    identifier: Some("RunMat:zoom:InvalidArgument"),
    when: "Argument count, target handle, factor, option, or zoom property is invalid.",
    message: "zoom: invalid argument",
};

pub const ZOOM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZOOM.INTERNAL",
    identifier: Some("RunMat:zoom:Internal"),
    when: "Internal zoom state or handle-object update fails.",
    message: "zoom: internal operation failed",
};

pub const ZOOM_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [ZOOM_INTEGER_TARGET_EXTENSION];

pub const ZOOM_INTEGER_TARGET_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "zoom-integer-target-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow a typed-integer alias for an encoded figure or axes handle",
    error_identifier: Some("RunMat:compatibility:ZoomIntegerTargetHandleExtension"),
};

pub const ZOOM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "zoom(integer_factor)",
        inputs: &ZOOM_INTEGER_FACTOR_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The positive finite factor crosses the graphics scaling boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "zoom(integer_target, option_or_factor)",
        inputs: &ZOOM_INTEGER_TARGET_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Strict mode rejects the encoded-handle alias before graphics state access.",
    },
];

pub const ZOOM_INTEGER_FACTOR_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "factor",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented positive numeric zoom factor admits positive integer scalars.",
    }];

pub const ZOOM_INTEGER_TARGET_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "target",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer aliases for RunMat's encoded graphics handles are separately gated.",
    }];
