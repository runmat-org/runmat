use crate::*;

const TRAPZ_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Trapezoidal integral output.",
}];

const TRAPZ_INPUTS_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample values.",
}];

const TRAPZ_INPUTS_Y_DIM: [BuiltinParamDescriptor; 2] = [
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

const TRAPZ_INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [
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

const TRAPZ_INPUTS_X_Y_DIM: [BuiltinParamDescriptor; 3] = [
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

const TRAPZ_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "Q = trapz(Y)",
        inputs: &TRAPZ_INPUTS_Y,
        outputs: &TRAPZ_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = trapz(Y, dim)",
        inputs: &TRAPZ_INPUTS_Y_DIM,
        outputs: &TRAPZ_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = trapz(X, Y)",
        inputs: &TRAPZ_INPUTS_X_Y,
        outputs: &TRAPZ_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = trapz(X, Y, dim)",
        inputs: &TRAPZ_INPUTS_X_Y_DIM,
        outputs: &TRAPZ_OUTPUT,
    },
];

pub const TRAPZ_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRAPZ.INVALID_ARGUMENT",
    identifier: Some("RunMat:trapz:InvalidArgument"),
    when: "Input argument count, dimension selector, or spacing arguments are invalid.",
    message: "trapz: invalid argument",
};

pub const TRAPZ_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRAPZ.INVALID_INPUT",
    identifier: Some("RunMat:trapz:InvalidInput"),
    when: "Input values cannot be converted to supported numeric integration domains.",
    message: "trapz: invalid input",
};

pub const TRAPZ_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRAPZ.INTERNAL",
    identifier: Some("RunMat:trapz:Internal"),
    when: "Integration execution fails during gather, allocation, or provider promotion.",
    message: "trapz: internal integration failure",
};

pub const TRAPZ_ERRORS: [BuiltinErrorDescriptor; 4] = [
    TRAPZ_ERROR_INVALID_ARGUMENT,
    TRAPZ_ERROR_INVALID_INPUT,
    TRAPZ_ERROR_INTERNAL,
    TRAPZ_ERROR_TOO_MANY_OUTPUTS,
];

pub const TRAPZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRAPZ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TRAPZ_ERRORS,
};

pub const TRAPZ_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "trapz-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "trapz with native typed-integer sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TrapzIntegerDataExtension"),
};

pub const TRAPZ_INTEGER_SPACING_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "trapz-integer-spacing",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "trapz with native typed-integer point spacing is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TrapzIntegerSpacingExtension"),
    };

pub const TRAPZ_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    TRAPZ_INTEGER_DATA_EXTENSION,
    TRAPZ_INTEGER_SPACING_EXTENSION,
];

const TRAPZ_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented sample-data domain is single or double. RunMat mode admits exact typed integers at a checked binary64 integration boundary.",
    }];

const TRAPZ_INTEGER_SPACING_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented spacing domain is single or double. RunMat mode admits exact typed integer scalar or vector spacing at a checked binary64 boundary.",
    }];

const TRAPZ_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented positive integer scalar dimension is decoded exactly from native integer storage and must fit the runtime index range.",
    }];

pub const TRAPZ_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = trapz(integer_Y)",
        inputs: &TRAPZ_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Compatibility admission and exactness validation precede provider dispatch. The trapezoidal sum and result use binary64.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = trapz(integer_X, Y)",
        inputs: &TRAPZ_INTEGER_SPACING_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer coordinates remain exact until checked binary64 conversion; result shape and floating class follow the sampled-data path.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = trapz(..., integer_dim)",
        inputs: &TRAPZ_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The dimension selects a reduction axis without entering floating arithmetic.",
    },
];

pub const TRAPZ_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRAPZ.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:trapz:TooManyOutputs"),
    when: "More outputs are requested than this public form returns.",
    message: "trapz: too many output arguments",
};
