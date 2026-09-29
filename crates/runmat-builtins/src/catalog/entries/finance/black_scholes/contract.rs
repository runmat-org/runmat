use crate::*;

pub const BLSPRICE_NONDOUBLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "blsprice-nondouble-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "non-double numeric blsprice inputs are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BlspriceNondoubleInputExtension"),
};

pub const BLSIMPV_NONDOUBLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "blsimpv-nondouble-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "non-double numeric blsimpv inputs are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BlsimpvNondoubleInputExtension"),
};

pub const BLSPRICE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [BLSPRICE_NONDOUBLE_EXTENSION];
pub const BLSIMPV_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [BLSIMPV_NONDOUBLE_EXTENSION];

const INTEGER_PRICING_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Price, Strike, Rate, Time, Volatility, Value, or Yield",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Native integer pricing values are separately gated and converted at the binary64 pricing boundary; wide values can round.",
    }];

pub const BLSPRICE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[Call, Put] = blsprice(integer_pricing_input, ...)",
        inputs: &INTEGER_PRICING_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer arguments use RunMat-only admission before provider work. The host pricing formula computes in double; a supported resident double provider may accelerate other input forms.",
    },
];

pub const BLSIMPV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Volatility = blsimpv(integer_pricing_input, ...)",
        inputs: &INTEGER_PRICING_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer arguments use RunMat-only admission before provider work. The host root finder computes in double and can round wide integer inputs.",
    },
];

const OUTPUT_CALL_PUT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Call",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "European call option price.",
    },
    BuiltinParamDescriptor {
        name: "Put",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "European put option price.",
    },
];

const OUTPUT_VOL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Volatility",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Implied annualized asset volatility.",
}];

const BLSPRICE_INPUTS: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "Price",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Current underlying asset price.",
    },
    BuiltinParamDescriptor {
        name: "Strike",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exercise price.",
    },
    BuiltinParamDescriptor {
        name: "Rate",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Continuously compounded risk-free rate.",
    },
    BuiltinParamDescriptor {
        name: "Time",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Time to expiration in years.",
    },
    BuiltinParamDescriptor {
        name: "Volatility",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Annualized asset volatility.",
    },
    BuiltinParamDescriptor {
        name: "Yield",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Continuously compounded yield.",
    },
];

const BLSIMPV_INPUTS: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "Price",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Current underlying asset price.",
    },
    BuiltinParamDescriptor {
        name: "Strike",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exercise price.",
    },
    BuiltinParamDescriptor {
        name: "Rate",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Continuously compounded risk-free rate.",
    },
    BuiltinParamDescriptor {
        name: "Time",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Time to expiration in years.",
    },
    BuiltinParamDescriptor {
        name: "Value",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Observed option value.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options Limit, Yield, Tolerance, Class, and Method.",
    },
];

const BLSPRICE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[Call, Put] = blsprice(Price, Strike, Rate, Time, Volatility)",
        inputs: &BLSPRICE_INPUTS,
        outputs: &OUTPUT_CALL_PUT,
    },
    BuiltinSignatureDescriptor {
        label: "[Call, Put] = blsprice(Price, Strike, Rate, Time, Volatility, Yield)",
        inputs: &BLSPRICE_INPUTS,
        outputs: &OUTPUT_CALL_PUT,
    },
];

const BLSIMPV_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Volatility = blsimpv(Price, Strike, Rate, Time, Value)",
        inputs: &BLSIMPV_INPUTS,
        outputs: &OUTPUT_VOL,
    },
    BuiltinSignatureDescriptor {
        label: "Volatility = blsimpv(Price, Strike, Rate, Time, Value, Name, Value)",
        inputs: &BLSIMPV_INPUTS,
        outputs: &OUTPUT_VOL,
    },
];

pub const BLACK_SCHOLES_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACK_SCHOLES.INVALID_ARGUMENT",
    identifier: Some("RunMat:BlackScholes:InvalidArgument"),
    when: "Inputs, option values, or requested output count are invalid.",
    message: "Black-Scholes builtin: invalid argument",
};

pub const BLACK_SCHOLES_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACK_SCHOLES.INVALID_INPUT",
    identifier: Some("RunMat:BlackScholes:InvalidInput"),
    when: "Inputs are nonnumeric, nonfinite, or have incompatible shapes.",
    message: "Black-Scholes builtin: invalid input",
};

pub const BLACK_SCHOLES_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACK_SCHOLES.INTERNAL",
    identifier: Some("RunMat:BlackScholes:Internal"),
    when: "Runtime cannot materialize the requested output.",
    message: "Black-Scholes builtin: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    BLACK_SCHOLES_ERROR_INVALID_ARGUMENT,
    BLACK_SCHOLES_ERROR_INVALID_INPUT,
    BLACK_SCHOLES_ERROR_INTERNAL,
];

pub const BLSPRICE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BLSPRICE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const BLSIMPV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BLSIMPV_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
