use crate::*;
pub const COLORMAP_OUTPUT_MAP: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "cmap",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current colormap as an m-by-3 normalized double RGB matrix.",
}];
pub const COLORMAP_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
pub const COLORMAP_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
pub const COLORMAP_INPUTS_NAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Colormap name.",
}];
pub const COLORMAP_INPUTS_RGB: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "map",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "m-by-3 single/double RGB map in [0,1], or uint8 RGB map in [0,255].",
}];
pub const COLORMAP_INPUTS_TARGET: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "target",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target figure or axes handle.",
}];
pub const COLORMAP_INPUTS_TARGET_MAP: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "target",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target figure or axes handle.",
    },
    BuiltinParamDescriptor {
        name: "map",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Colormap name or m-by-3 numeric RGB map.",
    },
];
pub const COLORMAP_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "cmap = colormap()",
        inputs: &COLORMAP_INPUTS_NONE,
        outputs: &COLORMAP_OUTPUT_MAP,
    },
    BuiltinSignatureDescriptor {
        label: "cmap = colormap(target)",
        inputs: &COLORMAP_INPUTS_TARGET,
        outputs: &COLORMAP_OUTPUT_MAP,
    },
    BuiltinSignatureDescriptor {
        label: "colormap(name)",
        inputs: &COLORMAP_INPUTS_NAME,
        outputs: &COLORMAP_OUTPUT_MAP,
    },
    BuiltinSignatureDescriptor {
        label: "colormap(map)",
        inputs: &COLORMAP_INPUTS_RGB,
        outputs: &COLORMAP_OUTPUT_MAP,
    },
    BuiltinSignatureDescriptor {
        label: "colormap(target, map)",
        inputs: &COLORMAP_INPUTS_TARGET_MAP,
        outputs: &COLORMAP_OUTPUT_MAP,
    },
];
pub const COLORMAP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORMAP.INVALID_ARGUMENT",
    identifier: Some("RunMat:colormap:InvalidArgument"),
    when: "Colormap input is missing, unknown, or not a valid m-by-3 RGB array.",
    message: "colormap: invalid argument",
};
pub const COLORMAP_ERRORS: [BuiltinErrorDescriptor; 2] =
    [COLORMAP_ERROR_INVALID_ARGUMENT, COLORMAP_ERROR_OUTPUTS];
pub const COLORMAP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COLORMAP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COLORMAP_ERRORS,
};

pub const COLORMAP_NON_UINT8_INTEGER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "colormap-non-uint8-integer-map",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "colormap with an integer RGB map other than uint8 is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ColormapNonUint8IntegerMapExtension"),
    };

pub const COLORMAP_AXES_DEFAULT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "colormap-axes-local-default",
    mode: BuiltinExtensionMode::ProfileDependentBehavior,
    description: "RunMat untargeted calls use the current axes palette; MATLAB profile uses the current figure palette. Explicit targets have the same meaning in either profile.",
    error_identifier: None,
};
pub const COLORMAP_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    COLORMAP_NON_UINT8_INTEGER_EXTENSION,
    COLORMAP_AXES_DEFAULT_EXTENSION,
];

pub const COLORMAP_UINT8_CLASSES: [BuiltinIntegerClass; 1] = [BuiltinIntegerClass::Uint8];
pub const COLORMAP_NON_UINT8_CLASSES: [BuiltinIntegerClass; 7] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint16,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];
pub const COLORMAP_UINT8_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "map",
        classes: &COLORMAP_UINT8_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "Public uint8 RGB maps use the full [0,255] channel range and are normalized to [0,1].",
    }];
pub const COLORMAP_NON_UINT8_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "map",
        classes: &COLORMAP_NON_UINT8_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode admits the other seven integer classes only when every RGB component is exactly 0 or 1.",
    }];

pub const COLORMAP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "colormap(uint8_map)",
        inputs: &COLORMAP_UINT8_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative uint8 channels are divided by 255 at the explicit host graphics-state boundary; the applied map is stored as normalized double and resident inputs remain unsupported.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "colormap(non_uint8_integer_map)",
        inputs: &COLORMAP_NON_UINT8_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "This useful RunMat-only form validates exact native 0/1 components before conversion to normalized host graphics metadata; resident inputs reject without provider access.",
    },
];
pub const COLORMAP_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORMAP.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:colormap:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "colormap: at most one output is supported",
};
