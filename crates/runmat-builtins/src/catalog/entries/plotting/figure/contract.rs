use crate::*;
const FIGURE_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fig",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat f64 graphics-handle encoding (not a MATLAB Figure object).",
}];

const FIGURE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const FIGURE_INPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat numeric figure identifier.",
}];

const FIGURE_INPUTS_PAIRS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "properties",
    ty: BuiltinParamType::PropertyName,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Figure property/value pairs such as 'Name', 'NumberTitle', 'Visible', 'Position', or 'Color'.",
}];

const FIGURE_INPUTS_HANDLE_PAIRS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "RunMat numeric figure identifier.",
    },
    BuiltinParamDescriptor {
        name: "properties",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Figure property/value pairs to apply after selecting or creating the figure.",
    },
];

const FIGURE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "fig = figure()",
        inputs: &FIGURE_INPUTS_NONE,
        outputs: &FIGURE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "fig = figure(h)",
        inputs: &FIGURE_INPUTS_HANDLE,
        outputs: &FIGURE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "fig = figure(property, value, ...)",
        inputs: &FIGURE_INPUTS_PAIRS,
        outputs: &FIGURE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "fig = figure(h, property, value, ...)",
        inputs: &FIGURE_INPUTS_HANDLE_PAIRS,
        outputs: &FIGURE_OUTPUT_HANDLE,
    },
];

pub const FIGURE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIGURE.INVALID_ARGUMENT",
    identifier: Some("RunMat:figure:InvalidArgument"),
    when: "Provided figure handle argument is invalid.",
    message: "figure: invalid argument",
};

const FIGURE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [FIGURE_ERROR_INVALID_ARGUMENT, FIGURE_ERROR_TOO_MANY_OUTPUTS];

pub const FIGURE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIGURE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FIGURE_ERRORS,
};

pub const FIGURE_INTEGER_TARGET_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "figure-integer-target",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "figure with a typed integer numeric target is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FigureIntegerTargetExtension"),
    };

pub const FIGURE_SINGLE_TARGET_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "figure-single-target",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "figure with a single-precision numeric target is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FigureSingleTargetExtension"),
};

pub const FIGURE_NEXT_SELECTOR_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "figure-next-selector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "figure('next') is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FigureNextSelectorExtension"),
};

pub const FIGURE_INTEGER_PROPERTY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "figure-integer-property-value",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "figure with typed integer property values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FigureIntegerPropertyValueExtension"),
    };

pub const FIGURE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FIGURE_INTEGER_TARGET_EXTENSION,
    FIGURE_SINGLE_TARGET_EXTENSION,
    FIGURE_NEXT_SELECTOR_EXTENSION,
    FIGURE_INTEGER_PROPERTY_EXTENSION,
];

const FIGURE_INTEGER_TARGET_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "h",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A host scalar from any built-in integer class is an exact RunMat numeric figure identifier and is independently compatibility gated. Logical and resident values are rejected.",
    }];

const FIGURE_INTEGER_POSITION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Position",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A four-element host integer vector is a RunMat-only geometry property value and must be exactly representable at the f64 figure-state boundary.",
    }];

const FIGURE_INTEGER_COLOR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Color",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A three-element host integer RGB vector is a RunMat-only property value. The normal [0,1] color constraint means admitted integer components are exactly zero or one.",
    }];

const FIGURE_INTEGER_SWITCH_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "state",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat-only NumberTitle and Visible integer scalars use an exact zero test; zero is off and every nonzero value is on without binary64 conversion.",
    }];

const FIGURE_INTEGER_OBJECT_PROPERTY_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "graphics_alias",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "CurrentAxes and SgTitle typed integers are RunMat-only aliases for the current f64 graphics registry and require exact binary64 representation before object lookup.",
    }];

pub const FIGURE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 6] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "fig = figure(integer_h, ...)",
        inputs: &FIGURE_INTEGER_TARGET_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "RunMat reads the authoritative host integer directly and requires the identifier to be positive and representable as u32. The returned f64 is RunMat's current opaque numeric graphics-handle encoding; MATLAB returns a Figure object, so this is a general graphics representation gap rather than numeric MATLAB equivalence.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fig = figure(..., 'Position', integer_position, ...)",
        inputs: &FIGURE_INTEGER_POSITION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "This independently gated RunMat extension reads authoritative integer storage, validates the four-element vector and exact f64 conversion before figure creation or selection, then stores host-double window geometry.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fig = figure(..., 'Color', integer_rgb, ...)",
        inputs: &FIGURE_INTEGER_COLOR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "This independently gated RunMat extension reads authoritative integer RGB storage before validation; the [0,1] range admits only exact zero/one components and renderer state is f32.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fig = figure(..., 'NumberTitle', integer_state, ...)",
        inputs: &FIGURE_INTEGER_SWITCH_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "This independently gated RunMat extension evaluates authoritative scalar integer zero/nonzero state without floating conversion before graphics mutation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fig = figure(..., 'Visible', integer_state, ...)",
        inputs: &FIGURE_INTEGER_SWITCH_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "This independently gated RunMat extension evaluates authoritative scalar integer zero/nonzero state without floating conversion before graphics mutation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "fig = figure(..., 'CurrentAxes'|'SgTitle', integer_graphics_alias, ...)",
        inputs: &FIGURE_INTEGER_OBJECT_PROPERTY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "These independently gated RunMat aliases require an exactly representable host integer before lookup in RunMat's f64 graphics-object registry; they do not make numeric aliases MATLAB graphics objects.",
    },
];

pub const FIGURE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIGURE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:figure:TooManyOutputs"),
    when: "More than 1 output arguments are requested.",
    message: "figure: too many output arguments",
};
