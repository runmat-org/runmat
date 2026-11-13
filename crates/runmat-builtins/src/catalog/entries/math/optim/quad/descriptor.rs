use crate::*;

const FUN: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "fun",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar integrand callback.",
};
const LOWER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "a",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Lower integration bound.",
};
const UPPER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Upper integration bound.",
};
const CORE: [BuiltinParamDescriptor; 3] = [FUN, LOWER, UPPER];
const WITH_OPTIONS: [BuiltinParamDescriptor; 6] = [
    FUN,
    LOWER,
    UPPER,
    BuiltinParamDescriptor {
        name: "tol",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1e-6"),
        description: "Positive absolute tolerance; empty selects the default.",
    },
    BuiltinParamDescriptor {
        name: "trace",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("false"),
        description: "Nonzero value prints [fcnEvals, a, b-a, Q] rows.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Arguments forwarded unchanged to the integrand.",
    },
];
const Q: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "q",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numerical integral estimate.",
}];
const Q_COUNT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "q",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerical integral estimate.",
    },
    BuiltinParamDescriptor {
        name: "fcnt",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integrand evaluation count.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "q = quad(fun, a, b)",
        inputs: &CORE,
        outputs: &Q,
    },
    BuiltinSignatureDescriptor {
        label: "q = quad(fun, a, b, tol, trace, p1, p2, ...)",
        inputs: &WITH_OPTIONS,
        outputs: &Q,
    },
    BuiltinSignatureDescriptor {
        label: "[q, fcnt] = quad(fun, a, b)",
        inputs: &CORE,
        outputs: &Q_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "[q, fcnt] = quad(fun, a, b, tol, trace, p1, p2, ...)",
        inputs: &WITH_OPTIONS,
        outputs: &Q_COUNT,
    },
];

pub const QUAD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUAD.INVALID_ARGUMENT",
    identifier: Some("RunMat:quad:InvalidArgument"),
    when: "Tolerance, trace, or argument grammar is invalid.",
    message: "quad: invalid argument",
};
pub const QUAD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUAD.INVALID_INPUT",
    identifier: Some("RunMat:quad:InvalidInput"),
    when: "Bounds, integrand values, or solver semantics are invalid.",
    message: "quad: invalid input",
};
pub const QUAD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUAD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:quad:TooManyOutputs"),
    when: "More than two outputs are requested.",
    message: "quad: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    QUAD_ERROR_INVALID_ARGUMENT,
    QUAD_ERROR_INVALID_INPUT,
    QUAD_ERROR_TOO_MANY_OUTPUTS,
];
pub const QUAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const QUAD_INTEGER_BOUND_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "quad-integer-bound",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "quad with native-class integer integration bounds is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QuadIntegerBoundExtension"),
};
pub const QUAD_INTEGER_TOLERANCE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "quad-integer-tolerance",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "quad with a native-class integer tolerance is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:QuadIntegerToleranceExtension"),
    };
pub const QUAD_INTEGER_CALLBACK_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "quad-integer-callback-result",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "quad with a native-class integer integrand result is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:QuadIntegerCallbackExtension"),
    };
pub const QUAD_LOGICAL_NUMERIC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "quad-logical-numeric",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "quad with logical bounds, tolerance, or integrand values is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QuadLogicalNumericExtension"),
};
pub const QUAD_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "quad-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "quad host fallback for explicit gpuArray values is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QuadResidentInputExtension"),
};
pub const QUAD_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    QUAD_INTEGER_BOUND_EXTENSION,
    QUAD_INTEGER_TOLERANCE_EXTENSION,
    QUAD_INTEGER_CALLBACK_EXTENSION,
    QUAD_LOGICAL_NUMERIC_EXTENSION,
    QUAD_RESIDENT_INPUT_EXTENSION,
];

const INTEGER_BOUND: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "a or b",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed bounds enter binary64 quadrature only after exact conversion.",
}];
const INTEGER_TOLERANCE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "tol",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed tolerance requires exact binary64 conversion.",
}];
const INTEGER_TRACE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "trace",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The trace toggle tests exact integer zero/nonzero state.",
}];
const INTEGER_PARAMETERS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "p1, p2, ...",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Forwarded callback payload retains its native integer storage.",
}];
const INTEGER_CALLBACK: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "fun result",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer callback result requires exact binary64 conversion.",
}];
pub const QUAD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "q = quad(fun, integer_a, integer_b, ___)",
        inputs: &INTEGER_BOUND,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only bounds are converted exactly before adaptive Simpson arithmetic.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "q = quad(fun, a, b, integer_tol, ___)",
        inputs: &INTEGER_TOLERANCE,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only tolerance is checked before floating arithmetic.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "q = quad(fun, a, b, tol, integer_trace, ___)",
        inputs: &INTEGER_TRACE,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Trace uses exact integer zero/nonzero state, without floating conversion.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "q = quad(fun, a, b, tol, trace, integer_p1, ___)",
        inputs: &INTEGER_PARAMETERS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Forwarded arguments preserve exact callback payload storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "q = quad(integer_returning_fun, a, b, ___)",
        inputs: &INTEGER_CALLBACK,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer callback result converts exactly before quadrature arithmetic.",
    },
];
