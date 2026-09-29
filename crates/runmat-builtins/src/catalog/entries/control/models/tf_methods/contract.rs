use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "New SISO transfer-function model.",
}];
const BINARY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "lhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "SISO tf model or scalar gain; at least one operand is a tf model in operator syntax.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "SISO tf model or scalar gain.",
    },
];
const UNARY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf model.",
}];
const POWER_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "sys",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "SISO tf model.",
    },
    BuiltinParamDescriptor {
        name: "exponent",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Finite integer-valued scalar within the signed 64-bit range.",
    },
];

// The model parser and polynomial arithmetic report these stable identifiers.
// Methods do not expose a fabricated, catch-all OperatorError identifier.
pub const TF_METHOD_ERROR_COEFFICIENT_CAPACITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.COEFFICIENT_CAPACITY",
    identifier: Some("RunMat:tf:CoefficientCapacity"),
    when: "A transfer-function power requires more polynomial coefficients than a host vector can represent.",
    message: "tf: power coefficient array exceeds host capacity",
};

// This ordered vocabulary owns each diagnostic once. An operator descriptor
// selects only the cases reachable through its parser and arithmetic path.
#[repr(usize)]
enum ErrorKind {
    InvalidArgument,
    InvalidModel,
    UnsupportedModel,
    InvalidCoefficients,
    DenominatorInvalid,
    InvalidSampleTime,
    InvalidVariable,
    SampleTimeMismatch,
    VariableMismatch,
    UnsupportedDelay,
    DivideByZero,
    InvalidExponent,
    CoefficientCapacity,
    Internal,
}

const ERRORS: [BuiltinErrorDescriptor; 14] = [
    BuiltinErrorDescriptor {
        code: "RM.TF.INVALID_ARGUMENT",
        identifier: Some("RunMat:tf:InvalidArgument"),
        when: "An operand is neither a tf model nor an admissible scalar.",
        message: "tf: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.TF.INVALID_MODEL",
        identifier: Some("RunMat:tf:InvalidModel"),
        when: "A required tf receiver is not a tf model object.",
        message: "tf: invalid model",
    },
    BuiltinErrorDescriptor {
        code: "RM.TF.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:tf:UnsupportedModel"),
        when: "The model class or property representation is unsupported.",
        message: "tf: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.TF.INVALID_COEFFICIENTS",
        identifier: Some("RunMat:tf:InvalidCoefficients"),
        when: "A model has malformed, nonfinite, or nonvector coefficients.",
        message: "tf: invalid coefficients",
    },
    BuiltinErrorDescriptor {
        code: "RM.TF.DENOMINATOR_INVALID",
        identifier: Some("RunMat:tf:DenominatorInvalid"),
        when: "The result or input model has a zero denominator.",
        message: "tf: invalid denominator coefficients",
    },
    BuiltinErrorDescriptor {
        code: "RM.TF.INVALID_SAMPLE_TIME",
        identifier: Some("RunMat:tf:InvalidSampleTime"),
        when: "A model has invalid sample-time metadata.",
        message: "tf: invalid sample time",
    },
    BuiltinErrorDescriptor {
        code: "RM.TF.INVALID_VARIABLE",
        identifier: Some("RunMat:tf:InvalidVariable"),
        when: "A model has invalid polynomial-variable metadata.",
        message: "tf: invalid variable",
    },
    BuiltinErrorDescriptor {
        code: "RM.TF.SAMPLE_TIME_MISMATCH",
        identifier: Some("RunMat:tf:SampleTimeMismatch"),
        when: "Binary model operands have different sample times.",
        message: "tf: sample times must match",
    },
    BuiltinErrorDescriptor {
        code: "RM.TF.VARIABLE_MISMATCH",
        identifier: Some("RunMat:tf:VariableMismatch"),
        when: "Binary model operands use different polynomial variables.",
        message: "tf: variables must match",
    },
    BuiltinErrorDescriptor {
        code: "RM.TF.UNSUPPORTED_DELAY",
        identifier: Some("RunMat:tf:UnsupportedDelay"),
        when: "Arithmetic involves nonzero input or output delay metadata.",
        message: "tf: delayed model arithmetic is unsupported",
    },
    BuiltinErrorDescriptor {
        code: "RM.TF.DIVIDE_BY_ZERO",
        identifier: Some("RunMat:tf:DivideByZero"),
        when: "Division or a negative power uses a zero transfer function as divisor.",
        message: "tf: cannot divide by a zero transfer function",
    },
    BuiltinErrorDescriptor {
        code: "RM.TF.INVALID_EXPONENT",
        identifier: Some("RunMat:tf:InvalidExponent"),
        when: "Power exponent is nonintegral, nonfinite, or outside the signed 64-bit range.",
        message: "tf: invalid exponent",
    },
    TF_METHOD_ERROR_COEFFICIENT_CAPACITY,
    BuiltinErrorDescriptor {
        code: "RM.TF.INTERNAL",
        identifier: Some("RunMat:tf:Internal"),
        when: "A result object or coefficient array cannot be constructed.",
        message: "tf: internal error",
    },
];

const fn error(kind: ErrorKind) -> BuiltinErrorDescriptor {
    let source = &ERRORS[kind as usize];
    BuiltinErrorDescriptor {
        code: source.code,
        identifier: source.identifier,
        when: source.when,
        message: source.message,
    }
}

macro_rules! errors {
    ($($kind:ident),+ $(,)?) => { &[$(error(ErrorKind::$kind)),+] };
}

const UNARY_ERRORS: &[BuiltinErrorDescriptor] = errors!(
    InvalidArgument,
    InvalidModel,
    UnsupportedModel,
    InvalidCoefficients,
    InvalidSampleTime,
    InvalidVariable,
    Internal,
);
const BINARY_ERRORS: &[BuiltinErrorDescriptor] = errors!(
    InvalidArgument,
    InvalidModel,
    UnsupportedModel,
    InvalidCoefficients,
    DenominatorInvalid,
    InvalidSampleTime,
    InvalidVariable,
    SampleTimeMismatch,
    VariableMismatch,
    UnsupportedDelay,
    Internal,
);
const DIVISION_ERRORS: &[BuiltinErrorDescriptor] = errors!(
    InvalidArgument,
    InvalidModel,
    UnsupportedModel,
    InvalidCoefficients,
    DenominatorInvalid,
    InvalidSampleTime,
    InvalidVariable,
    SampleTimeMismatch,
    VariableMismatch,
    UnsupportedDelay,
    DivideByZero,
    Internal,
);
const POWER_ERRORS: &[BuiltinErrorDescriptor] = errors!(
    InvalidArgument,
    InvalidModel,
    UnsupportedModel,
    InvalidCoefficients,
    DenominatorInvalid,
    InvalidSampleTime,
    InvalidVariable,
    UnsupportedDelay,
    DivideByZero,
    InvalidExponent,
    CoefficientCapacity,
    Internal,
);

macro_rules! descriptor {
    ($name:ident, $label:literal, $inputs:ident, $errors:ident) => {
        pub const $name: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[BuiltinSignatureDescriptor {
                label: $label,
                inputs: &$inputs,
                outputs: &OUTPUT,
            }],
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::MethodOnly,
            errors: $errors,
        };
    };
}

descriptor!(
    PLUS_DESCRIPTOR,
    "sys = tf.plus(lhs, rhs)",
    BINARY_INPUTS,
    BINARY_ERRORS
);
descriptor!(
    MINUS_DESCRIPTOR,
    "sys = tf.minus(lhs, rhs)",
    BINARY_INPUTS,
    BINARY_ERRORS
);
descriptor!(
    UPLUS_DESCRIPTOR,
    "sys = tf.uplus(sys)",
    UNARY_INPUTS,
    UNARY_ERRORS
);
descriptor!(
    UMINUS_DESCRIPTOR,
    "sys = tf.uminus(sys)",
    UNARY_INPUTS,
    UNARY_ERRORS
);
descriptor!(
    TIMES_DESCRIPTOR,
    "sys = tf.times(lhs, rhs)",
    BINARY_INPUTS,
    BINARY_ERRORS
);
descriptor!(
    MTIMES_DESCRIPTOR,
    "sys = tf.mtimes(lhs, rhs)",
    BINARY_INPUTS,
    BINARY_ERRORS
);
descriptor!(
    RDIVIDE_DESCRIPTOR,
    "sys = tf.rdivide(lhs, rhs)",
    BINARY_INPUTS,
    DIVISION_ERRORS
);
descriptor!(
    MRDIVIDE_DESCRIPTOR,
    "sys = tf.mrdivide(lhs, rhs)",
    BINARY_INPUTS,
    DIVISION_ERRORS
);
descriptor!(
    LDIVIDE_DESCRIPTOR,
    "sys = tf.ldivide(lhs, rhs)",
    BINARY_INPUTS,
    DIVISION_ERRORS
);
descriptor!(
    MLDIVIDE_DESCRIPTOR,
    "sys = tf.mldivide(lhs, rhs)",
    BINARY_INPUTS,
    DIVISION_ERRORS
);
descriptor!(
    POWER_DESCRIPTOR,
    "sys = tf.power(sys, exponent)",
    POWER_INPUTS,
    POWER_ERRORS
);
descriptor!(
    MPOWER_DESCRIPTOR,
    "sys = tf.mpower(sys, exponent)",
    POWER_INPUTS,
    POWER_ERRORS
);

const INTEGER_GAIN_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "lhs", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "A typed integer scalar gain is admitted only in RunMat mode and converted to binary64 before tf polynomial arithmetic; wide values may round." },
    BuiltinIntegerInputCapability { name: "rhs", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The right scalar gain follows the same potentially rounding binary64 conversion; a tf object itself is parsed through the shared model owner." },
];
pub const INTEGER_GAIN_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "tf binary operator with typed-integer scalar gain",
    inputs: &INTEGER_GAIN_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::ScalarOnly,
    notes: "The result is a host tf object with double or complex-double coefficients. Scalar integer gains may round at the binary64 boundary; this is distinct from checked integer coefficient inputs to tf construction.",
}];
const INTEGER_EXPONENT_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "exponent",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "RunMat decodes a typed integer scalar exactly as i64. Unsigned values above i64::MAX are rejected; no binary64 detour is used. This records RunMat's accepted typed-class behavior without asserting independent cross-runtime class parity.",
}];
pub const INTEGER_EXPONENT_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "tf power with typed-integer scalar exponent",
    inputs: &INTEGER_EXPONENT_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::ScalarOnly,
    notes: "The exact exponent selects repeated polynomial multiplication or reciprocal. The result is a host tf object.",
}];

// Binary gains use the coefficient gate already present in this consumed-model
// set. Unary and power methods consume the same mutable tf properties.
pub const CONSUMED_MODEL_EXTENSIONS: [BuiltinExtensionDescriptor; 3] =
    super::super::tf::TF_CONSUMED_PROPERTY_EXTENSIONS;
