use crate::*;
const AXES_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Axes handle.",
}];

const AXES_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const AXES_INPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "target",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Existing axes handle to make current, or figure handle to parent new axes.",
}];

const AXES_INPUTS_PAIRS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "properties",
    ty: BuiltinParamType::PropertyName,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Axes property/value pairs. The construction-time Parent property may target a figure.",
}];

const AXES_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "ax = axes()",
        inputs: &AXES_INPUTS_NONE,
        outputs: &AXES_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ax = axes(target)",
        inputs: &AXES_INPUTS_HANDLE,
        outputs: &AXES_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ax = axes(property, value, ...)",
        inputs: &AXES_INPUTS_PAIRS,
        outputs: &AXES_OUTPUT_HANDLE,
    },
];

const AXES_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AXES.INVALID_ARGUMENT",
    identifier: Some("RunMat:axes:InvalidArgument"),
    when: "Unsupported handle or property/value arguments are provided.",
    message: "axes: invalid argument",
};

const AXES_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AXES.INTERNAL",
    identifier: Some("RunMat:axes:Internal"),
    when: "Internal axes state operation fails.",
    message: "axes: internal operation failed",
};

const AXES_ERRORS: [BuiltinErrorDescriptor; 3] = [
    AXES_ERROR_INVALID_ARGUMENT,
    AXES_ERROR_INTERNAL,
    AXES_ERROR_TOO_MANY_OUTPUTS,
];

pub const AXES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &AXES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &AXES_ERRORS,
};

const AXES_RULER_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "XTick/YTick/XLim/YLim/ZLim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Documented numeric tick and limit vectors accept every built-in integer class; supported values cross once into host graphics state.",
    }];
const AXES_ASPECT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "DataAspectRatio",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "DataAspectRatio explicitly accepts every built-in integer class as a positive three-element vector.",
    }];
const AXES_APPEARANCE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Position/FontSize/XTickLabelRotation/YTickLabelRotation",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The supported layout vector and numeric scalar appearance properties accept real numeric values, including integer classes.",
    }];
pub const AXES_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "ax = axes(..., ruler_property, integer_values)",
        inputs: &AXES_RULER_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer values remain authoritative through shape validation and then cross the explicit f64 graphics-state boundary. The result is an opaque axes object encoding, not integer data.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ax = axes(..., \"DataAspectRatio\", integer_ratio)",
        inputs: &AXES_ASPECT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The exact typed vector is validated as three positive finite values before one conversion into the client graphics domain.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ax = axes(..., appearance_property, integer_value)",
        inputs: &AXES_APPEARANCE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Supported integer layout and scalar appearance controls are validated on the host and converted once into graphics state; resident property values are not an interactive GPU-array surface.",
    },
];

pub const AXES_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AXES.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:axes:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "axes: too many output arguments",
};
