use crate::*;

pub const CUMTRAPZ_INTEGER_Y_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cumtrapz-integer-y",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cumtrapz with typed-integer sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CumtrapzIntegerYExtension"),
};

pub const CUMTRAPZ_LOGICAL_Y_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cumtrapz-logical-y",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cumtrapz with logical sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CumtrapzLogicalYExtension"),
};

pub const CUMTRAPZ_INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cumtrapz-integer-spacing",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cumtrapz with typed-integer spacing is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CumtrapzIntegerSpacingExtension"),
};

pub const CUMTRAPZ_LOGICAL_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cumtrapz-logical-spacing",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cumtrapz with logical spacing is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CumtrapzLogicalSpacingExtension"),
};

pub const CUMTRAPZ_INTEGER_DIM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cumtrapz-integer-dim",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cumtrapz with a typed-integer dimension is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CumtrapzIntegerDimExtension"),
};

pub const CUMTRAPZ_TENSOR_SPACING_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cumtrapz-tensor-spacing",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cumtrapz with same-size tensor spacing is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CumtrapzTensorSpacingExtension"),
    };

pub const CUMTRAPZ_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    CUMTRAPZ_INTEGER_Y_EXTENSION,
    CUMTRAPZ_LOGICAL_Y_EXTENSION,
    CUMTRAPZ_INTEGER_X_EXTENSION,
    CUMTRAPZ_LOGICAL_X_EXTENSION,
    CUMTRAPZ_INTEGER_DIM_EXTENSION,
    CUMTRAPZ_TENSOR_SPACING_EXTENSION,
];

const CUMTRAPZ_INTEGER_Y_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Y",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes cross an explicitly gated binary64 integration boundary after exact-representability validation.",
}];

const CUMTRAPZ_INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer spacing classes are independently gated and validated before binary64 coordinate differences are formed.",
}];

const CUMTRAPZ_INTEGER_DIM_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "dim",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The public contract accepts a positive integer-valued scalar but does not list typed integer classes; ordinary double dim remains documented.",
}];

pub const CUMTRAPZ_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = cumtrapz(integer_Y, dim?)",
        inputs: &CUMTRAPZ_INTEGER_Y_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer samples remain authoritative through validation, then produce shape-preserving double output; eligible resident fallback returns through the first resident owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = cumtrapz(integer_X, Y, dim?)",
        inputs: &CUMTRAPZ_INTEGER_X_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer coordinates are independently gated and cross the floating spacing boundary; Y controls shape and documented floating precision.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = cumtrapz(___, integer_dim)",
        inputs: &CUMTRAPZ_INTEGER_DIM_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed dim is read exactly and controls traversal only; it does not determine the numeric result class.",
    },
];

const CUMTRAPZ_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cumulative trapezoidal integral output.",
}];

const CUMTRAPZ_INPUTS_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample values.",
}];

const CUMTRAPZ_INPUTS_Y_DIM: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integration dimension.",
    },
];

const CUMTRAPZ_INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample points or spacing.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
];

const CUMTRAPZ_INPUTS_X_Y_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample points or spacing.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integration dimension.",
    },
];

const CUMTRAPZ_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "Q = cumtrapz(Y)",
        inputs: &CUMTRAPZ_INPUTS_Y,
        outputs: &CUMTRAPZ_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = cumtrapz(Y, dim)",
        inputs: &CUMTRAPZ_INPUTS_Y_DIM,
        outputs: &CUMTRAPZ_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = cumtrapz(X, Y)",
        inputs: &CUMTRAPZ_INPUTS_X_Y,
        outputs: &CUMTRAPZ_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = cumtrapz(X, Y, dim)",
        inputs: &CUMTRAPZ_INPUTS_X_Y_DIM,
        outputs: &CUMTRAPZ_OUTPUT,
    },
];

pub const CUMTRAPZ_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMTRAPZ.INVALID_ARGUMENT",
    identifier: Some("RunMat:cumtrapz:InvalidArgument"),
    when: "Input argument count, dimension selector, or spacing arguments are invalid.",
    message: "cumtrapz: invalid argument",
};

pub const CUMTRAPZ_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMTRAPZ.INVALID_INPUT",
    identifier: Some("RunMat:cumtrapz:InvalidInput"),
    when: "Input values cannot be converted to supported numeric integration domains.",
    message: "cumtrapz: invalid input",
};

pub const CUMTRAPZ_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMTRAPZ.INTERNAL",
    identifier: Some("RunMat:cumtrapz:Internal"),
    when: "Integration execution fails during gather, allocation, or provider promotion.",
    message: "cumtrapz: internal integration failure",
};

pub const CUMTRAPZ_ERRORS: [BuiltinErrorDescriptor; 4] = [
    CUMTRAPZ_ERROR_INVALID_ARGUMENT,
    CUMTRAPZ_ERROR_INVALID_INPUT,
    CUMTRAPZ_ERROR_INTERNAL,
    CUMTRAPZ_ERROR_TOO_MANY_OUTPUTS,
];

pub const CUMTRAPZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CUMTRAPZ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CUMTRAPZ_ERRORS,
};

pub const CUMTRAPZ_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMTRAPZ.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:cumtrapz:TooManyOutputs"),
    when: "More outputs are requested than this public form returns.",
    message: "cumtrapz: too many output arguments",
};
