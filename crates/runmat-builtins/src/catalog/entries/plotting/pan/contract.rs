use crate::*;

pub const PAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PAN_ERRORS,
};

pub const PAN_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "p = pan()",
        inputs: &PAN_INPUTS_NONE,
        outputs: &PAN_OUTPUT_OBJECT,
    },
    BuiltinSignatureDescriptor {
        label: "pan()",
        inputs: &PAN_INPUTS_NONE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = pan(option)",
        inputs: &PAN_INPUTS_OPTION,
        outputs: &PAN_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = pan(target, option)",
        inputs: &PAN_INPUTS_TARGET_OPTION,
        outputs: &PAN_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "p = pan(fig)",
        inputs: &PAN_INPUTS_FIGURE,
        outputs: &PAN_OUTPUT_OBJECT,
    },
];

pub const PAN_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

pub const PAN_OUTPUT_OBJECT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pan mode object for the current or specified figure.",
}];

pub const PAN_INPUTS_OPTION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pan option: on, off, toggle, xon, or yon.",
}];

pub const PAN_OUTPUT_STATUS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Returns \"ok\" when the pan command succeeds.",
}];

pub const PAN_INPUTS_TARGET_OPTION: [BuiltinParamDescriptor; 2] = [
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
        description: "Pan option.",
    },
];

pub const PAN_INPUTS_FIGURE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fig",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target figure handle.",
}];

pub const PAN_ERRORS: [BuiltinErrorDescriptor; 2] =
    [PAN_ERROR_INVALID_ARGUMENT, PAN_ERROR_INTERNAL];

pub const PAN_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAN.INVALID_ARGUMENT",
    identifier: Some("RunMat:pan:InvalidArgument"),
    when: "Argument count, target handle, option, or pan property is invalid.",
    message: "pan: invalid argument",
};

pub const PAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAN.INTERNAL",
    identifier: Some("RunMat:pan:Internal"),
    when: "Internal pan state or handle-object update fails.",
    message: "pan: internal operation failed",
};

pub const PAN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [PAN_INTEGER_TARGET_EXTENSION];

pub const PAN_INTEGER_TARGET_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pan-integer-graphics-target",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pan with a typed-integer numeric graphics-target alias is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PanIntegerGraphicsTargetExtension"),
};

pub const PAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "set(pan_object, 'Enable', integer_zero_or_one)",
        inputs: &PAN_INTEGER_ENABLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The exact scalar is interpreted only as an on/off state and stored as pan-mode metadata.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "pan(integer_target, option)",
        inputs: &PAN_INTEGER_TARGET_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The compatibility gate precedes handle decoding and any pan-state mutation; resident targets are not gathered.",
    },
];

pub const PAN_INTEGER_ENABLE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Enable",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "[integer-audit-open] the compatibility target documents numeric or logical scalar 1/0 for the Enable property but does not enumerate native numeric classes; RunMat accepts every exact integer class without floating conversion.",
    }];

pub const PAN_INTEGER_TARGET_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "target",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "MATLAB requires a Figure or Axes object; typed-integer numeric aliases are a gated RunMat graphics representation extension.",
    }];
