//! Fitted probability distribution objects and object-aware distribution methods.

use runmat_accelerate_api::{GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, NumericScalar, ObjectInstance, ResolveContext, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast;
use crate::builtins::common::random;
use crate::builtins::common::random_args::{extract_dims, keyword_of};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::elementwise::gammaln::gammaln_nonnegative_scalar;
use crate::builtins::stats::summary::distribution_math;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const FITDIST_NAME: &str = "fitdist";
const PDF_NAME: &str = "pdf";
const CDF_NAME: &str = "cdf";
const RANDOM_NAME: &str = "random";
const PROBABILITY_DISTRIBUTION_CLASS: &str = "ProbabilityDistribution";
const MIN_POSITIVE: f64 = 1.0e-12;

const OUTPUT_PD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "pd",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Fitted probability distribution object.",
};

const OUTPUT_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Distribution function values.",
};

const OUTPUT_R: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random samples.",
};

const INPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample data or evaluation points.",
};

const INPUT_DIST: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "distname",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Distribution name.",
};

const INPUT_PD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "pd",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "ProbabilityDistribution object returned by fitdist.",
};

const INPUT_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Probability values.",
};

const INPUT_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "NameValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options.",
};

const INPUT_CDF_PARAMS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A...D",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Named-distribution parameters, followed optionally by \"upper\".",
};

const INPUT_SZ: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Output size.",
};

const FITDIST_INPUTS: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_DIST];
const FITDIST_INPUTS_OPTIONS: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_DIST, INPUT_OPTIONS];
const FITDIST_OUTPUTS: [BuiltinParamDescriptor; 1] = [OUTPUT_PD];
const PDF_INPUTS: [BuiltinParamDescriptor; 2] = [INPUT_PD, INPUT_X];
const PDF_NAME_INPUTS: [BuiltinParamDescriptor; 3] = [INPUT_DIST, INPUT_X, INPUT_OPTIONS];
const CDF_INPUTS: [BuiltinParamDescriptor; 2] = [INPUT_PD, INPUT_X];
const CDF_NAME_INPUTS: [BuiltinParamDescriptor; 3] = [INPUT_DIST, INPUT_X, INPUT_CDF_PARAMS];
const ICDF_INPUTS: [BuiltinParamDescriptor; 2] = [INPUT_PD, INPUT_P];
const RANDOM_INPUTS: [BuiltinParamDescriptor; 1] = [INPUT_PD];
const RANDOM_INPUTS_SIZE: [BuiltinParamDescriptor; 2] = [INPUT_PD, INPUT_SZ];
const RANDOM_NAME_INPUTS: [BuiltinParamDescriptor; 2] = [INPUT_DIST, INPUT_OPTIONS];
const DIST_OUTPUTS: [BuiltinParamDescriptor; 1] = [OUTPUT_Y];
const RANDOM_OUTPUTS: [BuiltinParamDescriptor; 1] = [OUTPUT_R];

const FITDIST_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "pd = fitdist(x, distname)",
        inputs: &FITDIST_INPUTS,
        outputs: &FITDIST_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "pd = fitdist(x, distname, Name, Value)",
        inputs: &FITDIST_INPUTS_OPTIONS,
        outputs: &FITDIST_OUTPUTS,
    },
];

const PDF_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "y = pdf(pd, x)",
        inputs: &PDF_INPUTS,
        outputs: &DIST_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "y = pdf(distname, x, params)",
        inputs: &PDF_NAME_INPUTS,
        outputs: &DIST_OUTPUTS,
    },
];

const PDF_INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pdf-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pdf with typed-integer evaluation points is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PdfIntegerXExtension"),
};
const PDF_INTEGER_PARAMETER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pdf-integer-parameters",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pdf with typed-integer distribution parameters is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PdfIntegerParametersExtension"),
};
pub const PDF_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [PDF_INTEGER_X_EXTENSION, PDF_INTEGER_PARAMETER_EXTENSION];
const PDF_INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double evaluation points. RunMat gates typed integers before provider access and requires exact conversion at the floating density boundary.",
    }];
const PDF_INTEGER_PARAMETER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A...D",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double named-distribution parameters. RunMat independently gates typed parameters and rejects lossy wide values.",
    }];
pub const PDF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = pdf(pd, integer_x) or pdf(name, integer_x, A...)",
        inputs: &PDF_INTEGER_X_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed evaluation points are a RunMat-only floating-boundary extension; density or mass values are floating outputs, and resident fallback restores the selected floating output class through the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = pdf(name, x, integer_A...D)",
        inputs: &PDF_INTEGER_PARAMETER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed distribution parameters are independently gated and converted only after exactness is proved.",
    },
];

const CDF_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "p = cdf(pd, x)",
        inputs: &CDF_INPUTS,
        outputs: &DIST_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "p = cdf(distname, x, params)",
        inputs: &CDF_NAME_INPUTS,
        outputs: &DIST_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "p = cdf(pd, x, \"upper\")",
        inputs: &CDF_INPUTS,
        outputs: &DIST_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "p = cdf(distname, x, params, \"upper\")",
        inputs: &CDF_NAME_INPUTS,
        outputs: &DIST_OUTPUTS,
    },
];

const CDF_INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cdf-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cdf with typed-integer evaluation points is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CdfIntegerXExtension"),
};

const CDF_INTEGER_PARAMETER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cdf-integer-parameters",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cdf with typed-integer distribution parameters is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CdfIntegerParametersExtension"),
};

const CDF_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cdf-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cdf with logical numeric inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CdfLogicalInputExtension"),
};

pub const CDF_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    CDF_INTEGER_X_EXTENSION,
    CDF_INTEGER_PARAMETER_EXTENSION,
    CDF_LOGICAL_INPUT_EXTENSION,
];

const CDF_INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer evaluation points are gated before provider access and must be exactly representable at the floating CDF boundary.",
    }];

const CDF_INTEGER_PARAMETER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A...D",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer named-distribution parameters are independently gated and must be exactly representable at the floating CDF boundary.",
    }];

pub const CDF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "p = cdf(pd, x) or cdf(name, x, A...) with integer x",
        inputs: &CDF_INTEGER_X_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only typed-integer x values produce double probabilities unless a named-form single parameter makes the result single; resident fallback restores output to the first owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "p = cdf(name, x, A...) with integer distribution parameters",
        inputs: &CDF_INTEGER_PARAMETER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only typed-integer parameters preserve authoritative storage through admission and scalar expansion, then cross an exact binary64 boundary for evaluation.",
    },
];

pub(crate) const ICDF_OBJECT_SIGNATURE: BuiltinSignatureDescriptor = BuiltinSignatureDescriptor {
    label: "x = icdf(pd, p)",
    inputs: &ICDF_INPUTS,
    outputs: &DIST_OUTPUTS,
};

const RANDOM_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = random(pd)",
        inputs: &RANDOM_INPUTS,
        outputs: &RANDOM_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "r = random(pd, sz)",
        inputs: &RANDOM_INPUTS_SIZE,
        outputs: &RANDOM_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "r = random(distname, params, sz)",
        inputs: &RANDOM_NAME_INPUTS,
        outputs: &RANDOM_OUTPUTS,
    },
];

const RANDOM_INTEGER_PARAMETER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "random-integer-parameters",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "random with typed-integer distribution parameters is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RandomIntegerParametersExtension"),
};
const RANDOM_INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "random-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "random with typed-integer size controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RandomIntegerSizeExtension"),
};
pub const RANDOM_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    RANDOM_INTEGER_PARAMETER_EXTENSION,
    RANDOM_INTEGER_SIZE_EXTENSION,
];
const RANDOM_INTEGER_PARAMETER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A...D",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single/double named-distribution parameters; typed integers are gated before conversion and must be exactly representable as binary64.",
    }];
const RANDOM_INTEGER_SIZE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sz1...szN or sz",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents single/double size controls; RunMat typed sizes are exact structural values and do not cross the distribution computation boundary.",
    }];
pub const RANDOM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "R = random(name,integer_A...D,...)",
        inputs: &RANDOM_INTEGER_PARAMETER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each typed parameter is admitted independently and converted once after exactness validation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "R = random(pd,integer_sz) or random(name,A...D,integer_sz)",
        inputs: &RANDOM_INTEGER_SIZE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed sizes gate before provider access and are decoded from authoritative integer storage without floating conversion.",
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITDIST.INVALID_ARGUMENT",
    identifier: Some("RunMat:fitdist:InvalidArgument"),
    when: "Sample data, distribution name, options, or evaluation inputs are malformed.",
    message: "fitdist: invalid argument",
};

const ERROR_NUMERICAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITDIST.NUMERICAL",
    identifier: Some("RunMat:fitdist:Numerical"),
    when: "Distribution parameter estimation fails to converge or is ill-conditioned.",
    message: "fitdist: numerical failure",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITDIST.INTERNAL",
    identifier: Some("RunMat:fitdist:Internal"),
    when: "RunMat cannot construct distribution outputs.",
    message: "fitdist: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_NUMERICAL, ERROR_INTERNAL];

pub const FITDIST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FITDIST_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const FITDIST_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fitdist-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fitdist with typed-integer sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FitdistIntegerDataExtension"),
};
const FITDIST_INTEGER_FREQUENCY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitdist-integer-frequency",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fitdist with typed-integer Frequency storage is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FitdistIntegerFrequencyExtension"),
    };
const FITDIST_RESIDENT_FALLBACK_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fitdist-resident-fallback",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fitdist gathers resident inputs and returns a host distribution object",
        error_identifier: Some("RunMat:compatibility:FitdistResidentFallbackExtension"),
    };
pub const FITDIST_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FITDIST_INTEGER_DATA_EXTENSION,
    FITDIST_INTEGER_FREQUENCY_EXTENSION,
    FITDIST_RESIDENT_FALLBACK_EXTENSION,
];
const FITDIST_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents double sample data. RunMat mode admits all eight integer classes only when every observation is exactly representable at the binary64 fitting boundary.",
    }];
const FITDIST_INTEGER_FREQUENCY_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Frequency",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target requires nonnegative integer-valued single or double counts. RunMat mode additionally admits exact typed-integer count storage.",
    }];
pub const FITDIST_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "pd = fitdist(integer_x, distname, ...)",
        inputs: &FITDIST_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer observations are gated before provider access, gathered exactly when resident, and converted once to binary64; the returned host probability-distribution object has floating parameters.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "pd = fitdist(x, distname, Frequency=integer_counts)",
        inputs: &FITDIST_INTEGER_FREQUENCY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Typed counts are independently gated before provider access and must remain nonnegative, finite, integral, shape-matched, and exactly representable as binary64 weights.",
    },
];

pub const PDF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PDF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const CDF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CDF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const RANDOM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RANDOM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DistributionKind {
    Normal,
    Exponential,
    Lognormal,
    Gamma,
    Weibull,
    Poisson,
}

impl DistributionKind {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Exponential => "Exponential",
            Self::Lognormal => "Lognormal",
            Self::Gamma => "Gamma",
            Self::Weibull => "Weibull",
            Self::Poisson => "Poisson",
        }
    }

    fn parameter_names(self) -> &'static [&'static str] {
        match self {
            Self::Normal => &["mu", "sigma"],
            Self::Exponential => &["mu"],
            Self::Lognormal => &["mu", "sigma"],
            Self::Gamma => &["a", "b"],
            Self::Weibull => &["a", "b"],
            Self::Poisson => &["lambda"],
        }
    }
}

#[derive(Clone, Debug)]
struct FittedDistribution {
    kind: DistributionKind,
    parameters: Vec<f64>,
    nlogl: f64,
    observations: f64,
}

#[derive(Clone, Debug)]
struct WeightedSample {
    values: Vec<f64>,
    weights: Vec<f64>,
    total_weight: f64,
}

#[derive(Default)]
struct FitOptions {
    frequency: Option<Vec<f64>>,
}

fn fitdist_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn distribution_eval_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.get(1) {
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn random_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    if args.len() <= 1 {
        Type::Num
    } else {
        Type::Tensor { shape: None }
    }
}

fn error_for(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if builtin == FITDIST_NAME {
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
    } else {
        let suffix = if std::ptr::eq(descriptor, &ERROR_INTERNAL) {
            "Internal"
        } else if std::ptr::eq(descriptor, &ERROR_NUMERICAL) {
            "Numerical"
        } else {
            "InvalidArgument"
        };
        builder = builder.with_identifier(format!("RunMat:{builtin}:{suffix}"));
    }
    builder.build()
}

fn error(descriptor: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(FITDIST_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    error(&ERROR_INVALID_ARGUMENT, message)
}

fn invalid_for(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    error_for(builtin, &ERROR_INVALID_ARGUMENT, message)
}

fn numerical(message: impl Into<String>) -> RuntimeError {
    error(&ERROR_NUMERICAL, message)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    error(&ERROR_INTERNAL, message)
}

fn internal_for(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    error_for(builtin, &ERROR_INTERNAL, message)
}

#[runtime_builtin(
    name = "fitdist",
    category = "stats/summary",
    summary = "Fit a probability distribution to sample data.",
    keywords = "fitdist,probability distribution,normal,exponential,lognormal,gamma,weibull,poisson,statistics",
    type_resolver(fitdist_type),
    descriptor(crate::builtins::stats::summary::fitdist::FITDIST_DESCRIPTOR),
    extensions(crate::builtins::stats::summary::fitdist::FITDIST_EXTENSIONS),
    integer_capabilities(crate::builtins::stats::summary::fitdist::FITDIST_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::fitdist"
)]
pub(crate) async fn fitdist_builtin(
    data: Value,
    distname: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    ensure_fitdist_integer_extensions(&data, &rest)?;
    ensure_fitdist_resident_fallback_extension(&data, &rest)?;
    let kind = parse_distribution_name(&distname)?;
    let sample_tensor = fitdist_value_to_tensor(data, "x").await?;
    let sample = parse_sample(sample_tensor, parse_fit_options(rest).await?)?;
    let fit = fit_distribution(kind, &sample)?;
    Ok(Value::Object(distribution_object(&fit)?))
}

#[runtime_builtin(
    name = "pdf",
    category = "stats/summary",
    summary = "Evaluate a fitted probability distribution density or mass function.",
    keywords = "pdf,fitdist,probability distribution,density,mass,statistics",
    type_resolver(distribution_eval_type),
    descriptor(crate::builtins::stats::summary::fitdist::PDF_DESCRIPTOR),
    extensions(crate::builtins::stats::summary::fitdist::PDF_EXTENSIONS),
    integer_capabilities(crate::builtins::stats::summary::fitdist::PDF_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::fitdist"
)]
pub(crate) async fn pdf_builtin(
    distribution: Value,
    x: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    ensure_pdf_integer_extensions(&distribution, &x, &rest).await?;
    evaluate_pdf(distribution, x, rest).await
}

async fn ensure_pdf_integer_extensions(
    distribution: &Value,
    input: &Value,
    parameters: &[Value],
) -> BuiltinResult<()> {
    if is_typed_integer_value(input) {
        crate::compatibility::ensure_builtin_extension_enabled(&PDF_INTEGER_X_EXTENSION, PDF_NAME)?;
        ensure_exact_pdf_integer_boundary(input, "x").await?;
    }
    if !matches!(distribution, Value::Object(_)) {
        for parameter in parameters
            .iter()
            .filter(|value| is_typed_integer_value(value))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &PDF_INTEGER_PARAMETER_EXTENSION,
                PDF_NAME,
            )?;
            ensure_exact_pdf_integer_boundary(parameter, "distribution parameter").await?;
        }
    }
    Ok(())
}

async fn ensure_exact_pdf_integer_boundary(value: &Value, role: &str) -> BuiltinResult<()> {
    if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value).await? {
        return Err(invalid_for(
            PDF_NAME,
            format!("pdf: integer {role} values must be exactly representable as double"),
        ));
    }
    Ok(())
}

async fn evaluate_pdf(distribution: Value, input: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if matches!(distribution, Value::Object(_)) {
        if !rest.is_empty() {
            return Err(invalid_for(
                PDF_NAME,
                "pdf: fitted distribution object form accepts exactly two inputs",
            ));
        }
        let precision = cdf_output_precision(&[&input]);
        let gpu_source = first_gpu_source(&[&input]);
        let fit = distribution_from_value(&distribution)?;
        let x = value_to_tensor(PDF_NAME, input).await?;
        let shape = x.shape.clone();
        let data = tensor::tensor_into_values_f64(x)
            .into_iter()
            .map(|value| pdf_scalar(&fit, value))
            .collect();
        return finish_pdf(shape, data, precision, gpu_source);
    }

    let kind = parse_distribution_name_for(PDF_NAME, &distribution)?;
    let mut original_inputs = Vec::with_capacity(rest.len() + 1);
    original_inputs.push(&input);
    original_inputs.extend(rest.iter());
    let precision = cdf_output_precision(&original_inputs);
    let gpu_source = first_gpu_source(&original_inputs);
    let x = value_to_tensor(PDF_NAME, input).await?;
    let parameters = parse_parameter_tensors(PDF_NAME, kind, rest).await?;
    let mut tensors = Vec::with_capacity(parameters.len() + 1);
    tensors.push(&x);
    tensors.extend(parameters.iter());
    let (mut values, shape) = broadcast_tensors_for(PDF_NAME, &tensors)?;
    let x_values = values.remove(0);
    let mut data = Vec::with_capacity(x_values.len());
    for index in 0..x_values.len() {
        let fit = FittedDistribution {
            kind,
            parameters: values.iter().map(|parameter| parameter[index]).collect(),
            nlogl: f64::NAN,
            observations: f64::NAN,
        };
        data.push(pdf_scalar(&fit, x_values[index]));
    }
    finish_pdf(shape, data, precision, gpu_source)
}

fn finish_pdf(
    shape: Vec<usize>,
    data: Vec<f64>,
    precision: CdfOutputPrecision,
    gpu_source: Option<GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let tensor = match precision {
        CdfOutputPrecision::Double => Tensor::new(data, shape),
        CdfOutputPrecision::Single => {
            Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
        }
    }
    .map_err(|error| internal_for(PDF_NAME, format!("pdf: {error}")))?;
    let output = match precision {
        CdfOutputPrecision::Double => tensor::tensor_into_value(tensor),
        CdfOutputPrecision::Single => Value::Tensor(tensor),
    };
    let Some(source) = gpu_source else {
        return Ok(output);
    };
    let restored = gpu_helpers::restore_class_preserving_value(&source, output, PDF_NAME)?;
    if runmat_accelerate_api::handle_is_explicit(&source)
        && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(invalid_for(
            PDF_NAME,
            "pdf: provider cannot preserve explicit gpuArray output residency and precision",
        ));
    }
    Ok(restored)
}

#[runtime_builtin(
    name = "cdf",
    category = "stats/summary",
    summary = "Evaluate a fitted probability distribution cumulative distribution function.",
    keywords = "cdf,fitdist,probability distribution,cumulative,statistics",
    type_resolver(distribution_eval_type),
    descriptor(crate::builtins::stats::summary::fitdist::CDF_DESCRIPTOR),
    extensions(crate::builtins::stats::summary::fitdist::CDF_EXTENSIONS),
    integer_capabilities(crate::builtins::stats::summary::fitdist::CDF_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::fitdist"
)]
pub(crate) async fn cdf_builtin(
    distribution: Value,
    x: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    evaluate_cdf(distribution, x, rest).await
}

#[derive(Clone, Copy)]
enum CdfOutputPrecision {
    Double,
    Single,
}

async fn evaluate_cdf(
    distribution: Value,
    input: Value,
    mut rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let upper = rest
        .last()
        .and_then(keyword_of)
        .is_some_and(|keyword| keyword.eq_ignore_ascii_case("upper"));
    if upper {
        rest.pop();
    }

    if matches!(distribution, Value::Object(_)) {
        if !rest.is_empty() {
            return Err(invalid_for(
                CDF_NAME,
                "cdf: fitted distribution object form accepts only x and optional 'upper'",
            ));
        }
        ensure_cdf_extensions(&input, &[])?;
        let precision = cdf_output_precision(&[&input]);
        let gpu_source = first_gpu_source(&[&input]);
        let fit = distribution_from_value(&distribution)?;
        let x = cdf_value_to_tensor(input).await?;
        ensure_exact_cdf_integer_boundary(&x, "x")?;
        let shape = x.shape.clone();
        let data = tensor::tensor_into_values_f64(x)
            .into_iter()
            .map(|value| cdf_scalar_with_tail(&fit, value, upper))
            .collect();
        return finish_cdf(shape, data, precision, gpu_source);
    }

    let kind = parse_distribution_name_for(CDF_NAME, &distribution)?;
    ensure_cdf_extensions(&input, &rest)?;
    let mut original_inputs = Vec::with_capacity(rest.len() + 1);
    original_inputs.push(&input);
    original_inputs.extend(rest.iter());
    let precision = cdf_output_precision(&original_inputs);
    let gpu_source = first_gpu_source(&original_inputs);
    let x = cdf_value_to_tensor(input).await?;
    ensure_exact_cdf_integer_boundary(&x, "x")?;
    let parameters = parse_cdf_parameter_tensors(kind, rest).await?;
    for (index, parameter) in parameters.iter().enumerate() {
        ensure_exact_cdf_integer_boundary(parameter, kind.parameter_names()[index])?;
    }
    let mut tensors = Vec::with_capacity(parameters.len() + 1);
    tensors.push(&x);
    tensors.extend(parameters.iter());
    let (mut values, shape) = broadcast_tensors_for(CDF_NAME, &tensors)?;
    let x_values = values.remove(0);
    let mut data = Vec::with_capacity(x_values.len());
    for index in 0..x_values.len() {
        let fit = FittedDistribution {
            kind,
            parameters: values.iter().map(|parameter| parameter[index]).collect(),
            nlogl: f64::NAN,
            observations: f64::NAN,
        };
        data.push(cdf_scalar_with_tail(&fit, x_values[index], upper));
    }
    finish_cdf(shape, data, precision, gpu_source)
}

fn ensure_cdf_extensions(input: &Value, parameters: &[Value]) -> BuiltinResult<()> {
    if is_typed_integer_value(input) {
        crate::compatibility::ensure_builtin_extension_enabled(&CDF_INTEGER_X_EXTENSION, CDF_NAME)?;
    }
    if parameters.iter().any(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CDF_INTEGER_PARAMETER_EXTENSION,
            CDF_NAME,
        )?;
    }
    if is_logical_value(input) || parameters.iter().any(is_logical_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CDF_LOGICAL_INPUT_EXTENSION,
            CDF_NAME,
        )?;
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn is_single_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle) == Some(ProviderPrecision::F32))
}

fn cdf_output_precision(inputs: &[&Value]) -> CdfOutputPrecision {
    if inputs.iter().any(|value| is_single_value(value)) {
        CdfOutputPrecision::Single
    } else {
        CdfOutputPrecision::Double
    }
}

fn first_gpu_source(inputs: &[&Value]) -> Option<GpuTensorHandle> {
    inputs.iter().find_map(|value| match value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    })
}

async fn cdf_value_to_tensor(value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_for(CDF_NAME, format!("cdf: {err}")))?;
    tensor::value_into_tensor_for(CDF_NAME, gathered)
        .map_err(|_| invalid_for(CDF_NAME, "cdf: expected numeric input"))
}

async fn parse_cdf_parameter_tensors(
    kind: DistributionKind,
    rest: Vec<Value>,
) -> BuiltinResult<Vec<Tensor>> {
    match kind {
        DistributionKind::Normal | DistributionKind::Lognormal => match rest.as_slice() {
            [] => Ok(vec![scalar_tensor(0.0), scalar_tensor(1.0)]),
            [mu] => Ok(vec![
                cdf_value_to_tensor(mu.clone()).await?,
                scalar_tensor(1.0),
            ]),
            [mu, sigma] => Ok(vec![
                cdf_value_to_tensor(mu.clone()).await?,
                cdf_value_to_tensor(sigma.clone()).await?,
            ]),
            _ => Err(invalid_for(
                CDF_NAME,
                format!(
                    "cdf: {} distribution expects x, x and mu, or x, mu, sigma",
                    kind.canonical_name()
                ),
            )),
        },
        DistributionKind::Exponential | DistributionKind::Poisson => {
            if rest.len() != 1 {
                return Err(invalid_for(
                    CDF_NAME,
                    format!(
                        "cdf: {} distribution expects one parameter",
                        kind.canonical_name()
                    ),
                ));
            }
            Ok(vec![cdf_value_to_tensor(rest[0].clone()).await?])
        }
        DistributionKind::Gamma | DistributionKind::Weibull => {
            if rest.len() != 2 {
                return Err(invalid_for(
                    CDF_NAME,
                    format!(
                        "cdf: {} distribution expects two parameters",
                        kind.canonical_name()
                    ),
                ));
            }
            Ok(vec![
                cdf_value_to_tensor(rest[0].clone()).await?,
                cdf_value_to_tensor(rest[1].clone()).await?,
            ])
        }
    }
}

fn ensure_exact_cdf_integer_boundary(tensor: &Tensor, name: &str) -> BuiltinResult<()> {
    if tensor.integer_storage().is_none() {
        return Ok(());
    }
    const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
    for index in 0..tensor.len() {
        let exact = match tensor.numeric_value_at(index) {
            Some(NumericScalar::I8(value)) => i128::from(value),
            Some(NumericScalar::I16(value)) => i128::from(value),
            Some(NumericScalar::I32(value)) => i128::from(value),
            Some(NumericScalar::I64(value)) => i128::from(value),
            Some(NumericScalar::U8(value)) => i128::from(value),
            Some(NumericScalar::U16(value)) => i128::from(value),
            Some(NumericScalar::U32(value)) => i128::from(value),
            Some(NumericScalar::U64(value)) => i128::from(value),
            _ => continue,
        };
        if !(-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&exact) {
            return Err(invalid_for(
                CDF_NAME,
                format!("cdf: integer {name} values must be exactly representable as double"),
            ));
        }
    }
    Ok(())
}

fn finish_cdf(
    shape: Vec<usize>,
    data: Vec<f64>,
    precision: CdfOutputPrecision,
    gpu_source: Option<GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let tensor = match precision {
        CdfOutputPrecision::Double => Tensor::new(data, shape),
        CdfOutputPrecision::Single => {
            Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
        }
    }
    .map_err(|err| internal_for(CDF_NAME, format!("cdf: {err}")))?;
    if let Some(source) = gpu_source {
        let provider = runmat_accelerate_api::provider_for_handle(&source)
            .or_else(runmat_accelerate_api::provider)
            .ok_or_else(|| {
                invalid_for(
                    CDF_NAME,
                    "cdf: no acceleration provider registered for GPU output",
                )
            })?;
        let handle = gpu_helpers::upload_tensor(provider, &tensor)
            .map_err(|err| invalid_for(CDF_NAME, format!("cdf: {err}")))?;
        return Ok(gpu_helpers::resident_gpu_value(handle));
    }
    match precision {
        CdfOutputPrecision::Double => Ok(tensor::tensor_into_value(tensor)),
        CdfOutputPrecision::Single => Ok(Value::Tensor(tensor)),
    }
}

#[runtime_builtin(
    name = "random",
    category = "stats/random",
    summary = "Generate random samples from a fitted probability distribution.",
    keywords = "random,fitdist,probability distribution,statistics",
    type_resolver(random_type),
    descriptor(crate::builtins::stats::summary::fitdist::RANDOM_DESCRIPTOR),
    extensions(crate::builtins::stats::summary::fitdist::RANDOM_EXTENSIONS),
    integer_capabilities(crate::builtins::stats::summary::fitdist::RANDOM_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::fitdist"
)]
pub(crate) async fn random_builtin(distribution: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_random_extensions(&distribution, &rest).await?;
    let gpu_source = random_gpu_source(&distribution, &rest)?;
    let (fit, shape) = if matches!(distribution, Value::Object(_)) {
        (
            distribution_from_value(&distribution)?,
            parse_shape_args(&rest).await?,
        )
    } else {
        parse_named_random_args(distribution, rest).await?
    };
    let len = tensor::element_count(&shape);
    let data = random_samples(&fit, len)?;
    finish_random(shape, data, gpu_source)
}

fn random_gpu_source(
    distribution: &Value,
    rest: &[Value],
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let parameter_count = if matches!(distribution, Value::Object(_)) {
        0
    } else {
        parse_distribution_name_for(RANDOM_NAME, distribution)?
            .parameter_names()
            .len()
    };
    gpu_helpers::select_resident_output_source(
        rest.iter()
            .take(parameter_count)
            .filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
        RANDOM_NAME,
    )
}

fn finish_random(
    shape: Vec<usize>,
    data: Vec<f64>,
    gpu_source: Option<GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let tensor = Tensor::new(data, shape)
        .map_err(|err| internal_for(RANDOM_NAME, format!("random: {err}")))?;
    let Some(source) = gpu_source else {
        return Ok(tensor::tensor_into_value(tensor));
    };
    let restored =
        gpu_helpers::restore_class_preserving_value(&source, Value::Tensor(tensor), RANDOM_NAME)?;
    if runmat_accelerate_api::handle_is_explicit(&source)
        && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(internal_for(
            RANDOM_NAME,
            "random: provider cannot preserve explicit gpuArray output residency",
        ));
    }
    Ok(restored)
}

async fn ensure_random_extensions(distribution: &Value, rest: &[Value]) -> BuiltinResult<()> {
    for value in rest {
        crate::builtins::common::validation::reject_typed_complex_integer(value, RANDOM_NAME)?;
    }
    let parameter_count = if matches!(distribution, Value::Object(_)) {
        0
    } else {
        parse_distribution_name_for(RANDOM_NAME, distribution)?
            .parameter_names()
            .len()
    };
    for (index, value) in rest.iter().enumerate() {
        if index < parameter_count {
            crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                value,
                &RANDOM_INTEGER_PARAMETER_EXTENSION,
                RANDOM_NAME,
                "distribution parameter",
            )
            .await?;
        } else if crate::builtins::common::validation::value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &RANDOM_INTEGER_SIZE_EXTENSION,
                RANDOM_NAME,
            )?;
        }
    }
    Ok(())
}

pub(crate) async fn icdf_probability_distribution(
    distribution: Value,
    p: Value,
) -> BuiltinResult<Value> {
    let fit = distribution_from_value(&distribution)?;
    let x = value_to_tensor("icdf", p).await?;
    let shape = x.shape.clone();
    let data = tensor::tensor_into_values_f64(x)
        .into_iter()
        .map(|value| icdf_scalar(&fit, value))
        .collect::<Vec<_>>();
    finish_for("icdf", shape, data)
}

async fn parse_named_eval_parameters(
    builtin: &'static str,
    kind: DistributionKind,
    rest: Vec<Value>,
) -> BuiltinResult<Vec<f64>> {
    let tensors = parse_parameter_tensors(builtin, kind, rest).await?;
    let (values, _shape) = broadcast_tensors_for(builtin, &tensors.iter().collect::<Vec<_>>())?;
    if values.iter().all(|values| values.len() == 1) {
        return Ok(values.into_iter().map(|values| values[0]).collect());
    }
    Err(invalid_for(
        builtin,
        format!("{builtin}: named distribution parameters must be scalar for this overload"),
    ))
}

async fn parse_named_random_args(
    distribution: Value,
    rest: Vec<Value>,
) -> BuiltinResult<(FittedDistribution, Vec<usize>)> {
    let kind = parse_distribution_name_for(RANDOM_NAME, &distribution)?;
    let parameter_count = kind.parameter_names().len();
    if rest.len() < parameter_count {
        return Err(invalid_for(
            RANDOM_NAME,
            format!(
                "random: {} distribution requires {} parameter argument(s)",
                kind.canonical_name(),
                parameter_count
            ),
        ));
    }
    let parameter_values = rest[..parameter_count].to_vec();
    let shape_args = rest[parameter_count..].to_vec();
    let params = parse_named_eval_parameters(RANDOM_NAME, kind, parameter_values).await?;
    let fit = FittedDistribution {
        kind,
        parameters: params,
        nlogl: f64::NAN,
        observations: f64::NAN,
    };
    let shape = parse_shape_args(&shape_args).await?;
    Ok((fit, shape))
}

async fn parse_parameter_tensors(
    builtin: &'static str,
    kind: DistributionKind,
    rest: Vec<Value>,
) -> BuiltinResult<Vec<Tensor>> {
    match kind {
        DistributionKind::Normal | DistributionKind::Lognormal => match rest.as_slice() {
            [] => Ok(vec![scalar_tensor(0.0), scalar_tensor(1.0)]),
            [mu] => Ok(vec![
                value_to_tensor(builtin, mu.clone()).await?,
                scalar_tensor(1.0),
            ]),
            [mu, sigma] => Ok(vec![
                value_to_tensor(builtin, mu.clone()).await?,
                value_to_tensor(builtin, sigma.clone()).await?,
            ]),
            _ => Err(invalid_for(
                builtin,
                format!(
                    "{builtin}: {} distribution expects x, x and mu, or x, mu, sigma",
                    kind.canonical_name()
                ),
            )),
        },
        DistributionKind::Exponential | DistributionKind::Poisson => {
            if rest.len() != 1 {
                return Err(invalid_for(
                    builtin,
                    format!(
                        "{builtin}: {} distribution expects one parameter",
                        kind.canonical_name()
                    ),
                ));
            }
            Ok(vec![value_to_tensor(builtin, rest[0].clone()).await?])
        }
        DistributionKind::Gamma | DistributionKind::Weibull => {
            if rest.len() != 2 {
                return Err(invalid_for(
                    builtin,
                    format!(
                        "{builtin}: {} distribution expects two parameters",
                        kind.canonical_name()
                    ),
                ));
            }
            Ok(vec![
                value_to_tensor(builtin, rest[0].clone()).await?,
                value_to_tensor(builtin, rest[1].clone()).await?,
            ])
        }
    }
}

async fn parse_fit_options(rest: Vec<Value>) -> BuiltinResult<FitOptions> {
    if !rest.len().is_multiple_of(2) {
        return Err(invalid("fitdist: name-value options must be paired"));
    }
    let mut options = FitOptions::default();
    for pair in rest.chunks_exact(2) {
        let name = keyword_of(&pair[0])
            .ok_or_else(|| invalid("fitdist: option names must be text"))?
            .to_ascii_lowercase();
        match name.as_str() {
            "frequency" | "freq" => {
                let tensor = fitdist_value_to_tensor(pair[1].clone(), "Frequency").await?;
                options.frequency = Some(tensor::tensor_into_values_f64(tensor));
            }
            "censoring" | "censor" => {
                return Err(invalid(
                    "fitdist: Censoring is not supported for fitted distributions yet",
                ))
            }
            "options" | "by" => {
                return Err(invalid(format!(
                    "fitdist: option '{name}' is not supported yet"
                )))
            }
            other => return Err(invalid(format!("fitdist: unknown option '{other}'"))),
        }
    }
    Ok(options)
}

fn ensure_fitdist_integer_extensions(data: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if is_typed_integer_value(data) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FITDIST_INTEGER_DATA_EXTENSION,
            FITDIST_NAME,
        )?;
    }
    for pair in rest.chunks_exact(2) {
        if keyword_of(&pair[0]).is_some_and(|name| {
            name.eq_ignore_ascii_case("frequency") || name.eq_ignore_ascii_case("freq")
        }) && is_typed_integer_value(&pair[1])
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FITDIST_INTEGER_FREQUENCY_EXTENSION,
                FITDIST_NAME,
            )?;
        }
    }
    Ok(())
}

fn ensure_fitdist_resident_fallback_extension(data: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if matches!(data, Value::GpuTensor(_))
        || rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FITDIST_RESIDENT_FALLBACK_EXTENSION,
            FITDIST_NAME,
        )?;
    }
    Ok(())
}

async fn fitdist_value_to_tensor(value: Value, role: &str) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid(format!("fitdist: {err}")))?;
    let tensor = tensor::value_into_tensor_for(FITDIST_NAME, gathered)
        .map_err(|_| invalid("fitdist: expected numeric input"))?;
    ensure_exact_fitdist_integer_boundary(&tensor, role)?;
    tensor::integer_tensor_to_f64(tensor).map_err(|err| invalid(format!("fitdist: {err}")))
}

fn ensure_exact_fitdist_integer_boundary(tensor: &Tensor, role: &str) -> BuiltinResult<()> {
    if tensor.integer_storage().is_none() {
        return Ok(());
    }
    for integer in tensor
        .integer_storage()
        .expect("integer storage checked above")
        .exact_values()
    {
        if !crate::builtins::math::trigonometry::cos::integer_is_exact_f64(&integer) {
            return Err(invalid(format!(
                "fitdist: integer {role} values must be exactly representable as double"
            )));
        }
    }
    Ok(())
}

async fn value_to_tensor(name: &'static str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_for(name, format!("{name}: {err}")))?;
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|_| invalid_for(name, format!("{name}: expected numeric input")))?;
    tensor::integer_tensor_to_f64(tensor).map_err(|err| invalid_for(name, format!("{name}: {err}")))
}

fn scalar_tensor(value: f64) -> Tensor {
    Tensor::new(vec![value], vec![1, 1]).expect("scalar tensor shape is valid")
}

fn broadcast_tensors_for(
    builtin: &'static str,
    inputs: &[&Tensor],
) -> BuiltinResult<(Vec<Vec<f64>>, Vec<usize>)> {
    let Some(first) = inputs.first() else {
        return Ok((Vec::new(), vec![1, 1]));
    };
    let mut shape = first.shape.clone();
    for tensor in inputs.iter().skip(1) {
        shape = broadcast::broadcast_shapes(builtin, &shape, &tensor.shape)
            .map_err(|err| invalid_for(builtin, err))?;
    }
    let mut values = Vec::with_capacity(inputs.len());
    for tensor in inputs {
        values.push(broadcast_tensor_to(builtin, tensor, &shape)?);
    }
    Ok((values, shape))
}

fn broadcast_tensor_to(
    builtin: &'static str,
    tensor: &Tensor,
    out_shape: &[usize],
) -> BuiltinResult<Vec<f64>> {
    let len = tensor::element_count(out_shape);
    if len == 0 {
        return Ok(Vec::new());
    }
    let in_shape = broadcast::align_shape(&tensor.shape, out_shape.len());
    let strides = broadcast::compute_strides(&in_shape);
    let mut out = Vec::with_capacity(len);
    let input_len = tensor::tensor_element_len(tensor);
    for idx in 0..len {
        let source_idx = broadcast::broadcast_index(idx, out_shape, &in_shape, &strides);
        if source_idx >= input_len {
            return Err(invalid_for(
                builtin,
                format!("{builtin}: tensor data does not match tensor shape"),
            ));
        }
        out.push(tensor::tensor_value_f64(tensor, source_idx));
    }
    Ok(out)
}

fn parse_sample(tensor: Tensor, options: FitOptions) -> BuiltinResult<WeightedSample> {
    if tensor.shape.iter().copied().filter(|dim| *dim > 1).count() > 1 {
        return Err(invalid("fitdist: data must be a vector"));
    }
    let sample_len = tensor::tensor_element_len(&tensor);
    let frequency = options.frequency.unwrap_or_else(|| vec![1.0; sample_len]);
    if frequency.len() != sample_len {
        return Err(invalid(
            "fitdist: Frequency must contain one value per observation",
        ));
    }
    let mut values = Vec::new();
    let mut weights = Vec::new();
    let mut total_weight = 0.0;
    let sample_values = tensor::tensor_values_f64(&tensor);
    for (value, weight) in sample_values.into_iter().zip(frequency) {
        if weight.is_nan() || weight < 0.0 || weight.fract() != 0.0 {
            return Err(invalid(
                "fitdist: Frequency values must be nonnegative finite integer counts",
            ));
        }
        if !weight.is_finite() {
            return Err(invalid(
                "fitdist: Frequency values must be nonnegative finite integer counts",
            ));
        }
        if value.is_nan() || weight == 0.0 {
            continue;
        }
        if !value.is_finite() {
            return Err(invalid("fitdist: data must not contain Inf values"));
        }
        values.push(value);
        weights.push(weight);
        total_weight += weight;
    }
    if values.is_empty() || total_weight <= 0.0 {
        return Err(invalid(
            "fitdist: at least one finite observation is required",
        ));
    }
    Ok(WeightedSample {
        values,
        weights,
        total_weight,
    })
}

fn fit_distribution(
    kind: DistributionKind,
    sample: &WeightedSample,
) -> BuiltinResult<FittedDistribution> {
    let parameters = match kind {
        DistributionKind::Normal => fit_normal(sample)?,
        DistributionKind::Exponential => fit_exponential(sample)?,
        DistributionKind::Lognormal => fit_lognormal(sample)?,
        DistributionKind::Gamma => fit_gamma(sample)?,
        DistributionKind::Weibull => fit_weibull(sample)?,
        DistributionKind::Poisson => fit_poisson(sample)?,
    };
    let fit = FittedDistribution {
        kind,
        nlogl: sample
            .values
            .iter()
            .zip(sample.weights.iter())
            .map(|(value, weight)| -weight * pdf_scalar_raw(kind, &parameters, *value).ln())
            .sum(),
        parameters,
        observations: sample.total_weight,
    };
    Ok(fit)
}

fn fit_normal(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    let mu = weighted_mean(sample);
    let variance = sample
        .values
        .iter()
        .zip(sample.weights.iter())
        .map(|(value, weight)| weight * (value - mu).powi(2))
        .sum::<f64>()
        / sample.total_weight;
    if variance < 0.0 {
        return Err(numerical("fitdist: normal variance is invalid"));
    }
    Ok(vec![mu, variance.sqrt()])
}

fn fit_exponential(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    require_range(sample, |value| value >= 0.0, "Exponential")?;
    let mu = weighted_mean(sample);
    if mu <= 0.0 {
        return Err(invalid("fitdist: Exponential data must have positive mean"));
    }
    Ok(vec![mu])
}

fn fit_lognormal(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    require_range(sample, |value| value > 0.0, "Lognormal")?;
    let logs = transformed_sample(sample, |value| value.ln());
    fit_normal(&logs)
}

fn fit_gamma(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    require_range(sample, |value| value > 0.0, "Gamma")?;
    let mean = weighted_mean(sample);
    let mean_log = sample
        .values
        .iter()
        .zip(sample.weights.iter())
        .map(|(value, weight)| weight * value.ln())
        .sum::<f64>()
        / sample.total_weight;
    let s = mean.ln() - mean_log;
    if s <= 0.0 {
        return Err(numerical(
            "fitdist: Gamma shape is undefined for nearly constant data",
        ));
    }
    let mut shape =
        ((3.0 - s + ((s - 3.0).powi(2) + 24.0 * s).sqrt()) / (12.0 * s)).max(MIN_POSITIVE);
    for _ in 0..64 {
        let f = shape.ln() - digamma(shape) - s;
        let fp = 1.0 / shape - trigamma(shape);
        let step = f / fp;
        let candidate = shape - step;
        if candidate.is_finite() && candidate > 0.0 {
            shape = candidate;
        } else {
            shape *= 0.5;
        }
        if step.abs() <= 1.0e-12 * shape.max(1.0) {
            break;
        }
    }
    if !shape.is_finite() || shape <= 0.0 {
        return Err(numerical("fitdist: Gamma fit did not converge"));
    }
    Ok(vec![shape, mean / shape])
}

fn fit_weibull(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    require_range(sample, |value| value > 0.0, "Weibull")?;
    let logs = transformed_sample(sample, |value| value.ln());
    let log_mean = weighted_mean(&logs);
    let log_var = logs
        .values
        .iter()
        .zip(logs.weights.iter())
        .map(|(value, weight)| weight * (value - log_mean).powi(2))
        .sum::<f64>()
        / logs.total_weight;
    let mut shape = (std::f64::consts::PI / (6.0 * log_var.max(1.0e-12)).sqrt()).clamp(0.1, 100.0);
    for _ in 0..80 {
        let (a, b, c) = weibull_sums(sample, shape);
        let g = a / b - log_mean - 1.0 / shape;
        let gp = c / b - (a / b).powi(2) + 1.0 / shape.powi(2);
        let step = g / gp;
        let candidate = shape - step;
        if candidate.is_finite() && candidate > 0.0 {
            shape = candidate;
        } else {
            shape *= 0.5;
        }
        if step.abs() <= 1.0e-11 * shape.max(1.0) {
            break;
        }
    }
    if !shape.is_finite() || shape <= 0.0 {
        return Err(numerical("fitdist: Weibull fit did not converge"));
    }
    let scale = (sample
        .values
        .iter()
        .zip(sample.weights.iter())
        .map(|(value, weight)| weight * value.powf(shape))
        .sum::<f64>()
        / sample.total_weight)
        .powf(1.0 / shape);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(numerical("fitdist: Weibull scale is invalid"));
    }
    Ok(vec![scale, shape])
}

fn fit_poisson(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    require_range(
        sample,
        |value| value >= 0.0 && value.fract() == 0.0,
        "Poisson",
    )?;
    Ok(vec![weighted_mean(sample)])
}

fn require_range(
    sample: &WeightedSample,
    pred: impl Fn(f64) -> bool,
    name: &str,
) -> BuiltinResult<()> {
    if sample.values.iter().copied().all(pred) {
        Ok(())
    } else {
        Err(invalid(format!(
            "fitdist: {name} distribution data are outside the supported range"
        )))
    }
}

fn transformed_sample(sample: &WeightedSample, transform: impl Fn(f64) -> f64) -> WeightedSample {
    WeightedSample {
        values: sample.values.iter().copied().map(transform).collect(),
        weights: sample.weights.clone(),
        total_weight: sample.total_weight,
    }
}

fn weighted_mean(sample: &WeightedSample) -> f64 {
    sample
        .values
        .iter()
        .zip(sample.weights.iter())
        .map(|(value, weight)| value * weight)
        .sum::<f64>()
        / sample.total_weight
}

fn weibull_sums(sample: &WeightedSample, shape: f64) -> (f64, f64, f64) {
    let mut weighted_xk_log = 0.0;
    let mut weighted_xk = 0.0;
    let mut weighted_xk_log2 = 0.0;
    for (value, weight) in sample.values.iter().zip(sample.weights.iter()) {
        let log_value = value.ln();
        let xk = value.powf(shape);
        weighted_xk_log += weight * xk * log_value;
        weighted_xk += weight * xk;
        weighted_xk_log2 += weight * xk * log_value * log_value;
    }
    (weighted_xk_log, weighted_xk, weighted_xk_log2)
}

fn distribution_object(fit: &FittedDistribution) -> BuiltinResult<ObjectInstance> {
    let mut object = ObjectInstance::new(PROBABILITY_DISTRIBUTION_CLASS.to_string());
    object.properties.insert(
        "DistributionName".to_string(),
        Value::String(fit.kind.canonical_name().to_string()),
    );
    object.properties.insert(
        "DistName".to_string(),
        Value::String(fit.kind.canonical_name().to_string()),
    );
    object.properties.insert(
        "ParameterNames".to_string(),
        Value::StringArray(string_row(
            fit.kind
                .parameter_names()
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
        )?),
    );
    object.properties.insert(
        "ParameterValues".to_string(),
        Value::Tensor(
            Tensor::new(fit.parameters.clone(), vec![1, fit.parameters.len()])
                .map_err(|err| internal(format!("fitdist: {err}")))?,
        ),
    );
    object.properties.insert(
        "NumParameters".to_string(),
        Value::Num(fit.parameters.len() as f64),
    );
    object
        .properties
        .insert("NumObservations".to_string(), Value::Num(fit.observations));
    object
        .properties
        .insert("NLogL".to_string(), Value::Num(fit.nlogl));
    object
        .properties
        .insert("IsTruncated".to_string(), Value::Bool(false));
    for (name, value) in fit.kind.parameter_names().iter().zip(fit.parameters.iter()) {
        object
            .properties
            .insert((*name).to_string(), Value::Num(*value));
    }
    Ok(object)
}

fn string_row(values: Vec<String>) -> BuiltinResult<StringArray> {
    StringArray::new(values.clone(), vec![1, values.len()])
        .map_err(|err| internal(format!("fitdist: {err}")))
}

fn distribution_from_value(value: &Value) -> BuiltinResult<FittedDistribution> {
    let Value::Object(object) = value else {
        return Err(invalid("fitdist: expected ProbabilityDistribution object"));
    };
    if !object.is_class(PROBABILITY_DISTRIBUTION_CLASS) {
        return Err(invalid(format!(
            "fitdist: expected ProbabilityDistribution object, got {}",
            object.class_name
        )));
    }
    let dist_name = string_property(object, "DistributionName")?;
    let kind = parse_distribution_keyword(&dist_name)?;
    let parameters = numeric_vector_property(object, "ParameterValues")?;
    if parameters.len() != kind.parameter_names().len() {
        return Err(invalid(
            "fitdist: ProbabilityDistribution object has malformed ParameterValues",
        ));
    }
    let nlogl = numeric_scalar_property(object, "NLogL").unwrap_or(f64::NAN);
    let observations = numeric_scalar_property(object, "NumObservations").unwrap_or(f64::NAN);
    Ok(FittedDistribution {
        kind,
        parameters,
        nlogl,
        observations,
    })
}

fn string_property(object: &ObjectInstance, name: &str) -> BuiltinResult<String> {
    match object.properties.get(name) {
        Some(Value::String(value)) => Ok(value.clone()),
        Some(Value::CharArray(chars)) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        _ => Err(invalid(format!(
            "fitdist: ProbabilityDistribution object is missing {name}"
        ))),
    }
}

fn numeric_vector_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<f64>> {
    match object.properties.get(name) {
        Some(Value::Tensor(tensor)) => Ok(tensor::tensor_values_f64(tensor)),
        Some(Value::Num(value)) => Ok(vec![*value]),
        _ => Err(invalid(format!(
            "fitdist: ProbabilityDistribution object is missing {name}"
        ))),
    }
}

fn numeric_scalar_property(object: &ObjectInstance, name: &str) -> Option<f64> {
    match object.properties.get(name) {
        Some(Value::Num(value)) => Some(*value),
        _ => None,
    }
}

fn parse_distribution_name(value: &Value) -> BuiltinResult<DistributionKind> {
    parse_distribution_name_for(FITDIST_NAME, value)
}

fn parse_distribution_name_for(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<DistributionKind> {
    let keyword = keyword_of(value).ok_or_else(|| {
        invalid_for(
            builtin,
            format!("{builtin}: distribution name must be a string scalar"),
        )
    })?;
    parse_distribution_keyword_for(builtin, &keyword)
}

fn parse_distribution_keyword(keyword: &str) -> BuiltinResult<DistributionKind> {
    parse_distribution_keyword_for(FITDIST_NAME, keyword)
}

fn parse_distribution_keyword_for(
    builtin: &'static str,
    keyword: &str,
) -> BuiltinResult<DistributionKind> {
    let normalized = keyword
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    match normalized.as_str() {
        "normal" | "norm" | "gaussian" => Ok(DistributionKind::Normal),
        "exponential" | "exp" => Ok(DistributionKind::Exponential),
        "lognormal" | "logn" => Ok(DistributionKind::Lognormal),
        "gamma" | "gam" => Ok(DistributionKind::Gamma),
        "weibull" | "wbl" => Ok(DistributionKind::Weibull),
        "poisson" | "poiss" => Ok(DistributionKind::Poisson),
        _ => Err(invalid_for(
            builtin,
            format!("{builtin}: unsupported distribution '{keyword}'"),
        )),
    }
}

fn pdf_scalar(fit: &FittedDistribution, x: f64) -> f64 {
    pdf_scalar_raw(fit.kind, &fit.parameters, x)
}

fn pdf_scalar_raw(kind: DistributionKind, params: &[f64], x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    match kind {
        DistributionKind::Normal => {
            let [mu, sigma] = two(params);
            if sigma <= 0.0 {
                return if x == mu { f64::INFINITY } else { 0.0 };
            }
            distribution_math::standard_normal_pdf((x - mu) / sigma) / sigma
        }
        DistributionKind::Exponential => {
            let mu = params[0];
            if x < 0.0 || mu <= 0.0 {
                0.0
            } else {
                (-x / mu).exp() / mu
            }
        }
        DistributionKind::Lognormal => {
            let [mu, sigma] = two(params);
            if x <= 0.0 || sigma <= 0.0 {
                0.0
            } else {
                distribution_math::standard_normal_pdf((x.ln() - mu) / sigma) / (x * sigma)
            }
        }
        DistributionKind::Gamma => {
            let [shape, scale] = two(params);
            if x < 0.0 || shape <= 0.0 || scale <= 0.0 {
                0.0
            } else if x == 0.0 && shape < 1.0 {
                f64::INFINITY
            } else if x == 0.0 && shape > 1.0 {
                0.0
            } else {
                ((shape - 1.0) * x.ln()
                    - x / scale
                    - gammaln_nonnegative_scalar(shape)
                    - shape * scale.ln())
                .exp()
            }
        }
        DistributionKind::Weibull => {
            let [scale, shape] = two(params);
            if x < 0.0 || scale <= 0.0 || shape <= 0.0 {
                0.0
            } else if x == 0.0 && shape < 1.0 {
                f64::INFINITY
            } else if x == 0.0 && shape > 1.0 {
                0.0
            } else {
                (shape / scale) * (x / scale).powf(shape - 1.0) * (-(x / scale).powf(shape)).exp()
            }
        }
        DistributionKind::Poisson => {
            let lambda = params[0];
            if x < 0.0 || x.fract() != 0.0 || lambda < 0.0 {
                0.0
            } else if lambda == 0.0 {
                if x == 0.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                (x * lambda.ln() - lambda - gammaln_nonnegative_scalar(x + 1.0)).exp()
            }
        }
    }
}

fn cdf_scalar_with_tail(fit: &FittedDistribution, x: f64, upper: bool) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    match fit.kind {
        DistributionKind::Normal => {
            let [mu, sigma] = two(&fit.parameters);
            if sigma <= 0.0 {
                let lower = if x < mu { 0.0 } else { 1.0 };
                if upper {
                    1.0 - lower
                } else {
                    lower
                }
            } else if upper {
                distribution_math::standard_normal_cdf((mu - x) / sigma)
            } else {
                distribution_math::standard_normal_cdf((x - mu) / sigma)
            }
        }
        DistributionKind::Exponential => {
            let mu = fit.parameters[0];
            if mu <= 0.0 {
                if upper {
                    1.0
                } else {
                    0.0
                }
            } else if x < 0.0 {
                if upper {
                    1.0
                } else {
                    0.0
                }
            } else if upper {
                (-x / mu).exp()
            } else {
                -(-x / mu).exp_m1()
            }
        }
        DistributionKind::Lognormal => {
            let [mu, sigma] = two(&fit.parameters);
            if x <= 0.0 || sigma <= 0.0 {
                if upper {
                    1.0
                } else {
                    0.0
                }
            } else if upper {
                distribution_math::standard_normal_cdf((mu - x.ln()) / sigma)
            } else {
                distribution_math::standard_normal_cdf((x.ln() - mu) / sigma)
            }
        }
        DistributionKind::Gamma => {
            let [shape, scale] = two(&fit.parameters);
            if x <= 0.0 || shape <= 0.0 || scale <= 0.0 {
                if upper {
                    1.0
                } else {
                    0.0
                }
            } else if upper {
                distribution_math::regularized_gamma_q(shape, x / scale)
            } else {
                distribution_math::regularized_gamma_p(shape, x / scale)
            }
        }
        DistributionKind::Weibull => {
            let [scale, shape] = two(&fit.parameters);
            if x < 0.0 || scale <= 0.0 || shape <= 0.0 {
                if upper {
                    1.0
                } else {
                    0.0
                }
            } else if upper {
                (-(x / scale).powf(shape)).exp()
            } else {
                -(-(x / scale).powf(shape)).exp_m1()
            }
        }
        DistributionKind::Poisson => {
            let lambda = fit.parameters[0];
            if x < 0.0 {
                if upper {
                    1.0
                } else {
                    0.0
                }
            } else if lambda == 0.0 {
                if upper {
                    0.0
                } else {
                    1.0
                }
            } else if upper {
                distribution_math::regularized_gamma_p(x.floor() + 1.0, lambda)
            } else {
                distribution_math::regularized_gamma_q(x.floor() + 1.0, lambda)
            }
        }
    }
}

fn icdf_scalar(fit: &FittedDistribution, p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    match fit.kind {
        DistributionKind::Normal => {
            let [mu, sigma] = two(&fit.parameters);
            mu + sigma * distribution_math::standard_normal_inv(p)
        }
        DistributionKind::Exponential => {
            let mu = fit.parameters[0];
            if p == 1.0 {
                f64::INFINITY
            } else {
                -mu * (1.0 - p).ln()
            }
        }
        DistributionKind::Lognormal => {
            let [mu, sigma] = two(&fit.parameters);
            (mu + sigma * distribution_math::standard_normal_inv(p)).exp()
        }
        DistributionKind::Gamma => {
            let [shape, scale] = two(&fit.parameters);
            invert_positive(p, shape * scale, |x| {
                distribution_math::regularized_gamma_p(shape, x / scale)
            })
        }
        DistributionKind::Weibull => {
            let [scale, shape] = two(&fit.parameters);
            if p == 1.0 {
                f64::INFINITY
            } else {
                scale * (-(1.0 - p).ln()).powf(1.0 / shape)
            }
        }
        DistributionKind::Poisson => poisson_inv(p, fit.parameters[0]),
    }
}

fn random_samples(fit: &FittedDistribution, len: usize) -> BuiltinResult<Vec<f64>> {
    match fit.kind {
        DistributionKind::Normal => {
            let [mu, sigma] = two(&fit.parameters);
            random::generate_normal_scaled(mu, sigma, len, RANDOM_NAME)
        }
        DistributionKind::Exponential => {
            random::generate_exponential(fit.parameters[0].max(MIN_POSITIVE), len, RANDOM_NAME)
        }
        DistributionKind::Lognormal => {
            let [mu, sigma] = two(&fit.parameters);
            random::generate_normal_scaled(mu, sigma, len, RANDOM_NAME)
                .map(|values| values.into_iter().map(f64::exp).collect())
        }
        DistributionKind::Gamma => {
            let [shape, scale] = two(&fit.parameters);
            random::generate_gamma_shape_scale(&[shape], &[scale], len, RANDOM_NAME)
        }
        DistributionKind::Weibull => {
            let [scale, shape] = two(&fit.parameters);
            random::generate_weibull(&[scale], &[shape], len, RANDOM_NAME)
        }
        DistributionKind::Poisson => {
            let uniforms = random::generate_uniform(len, RANDOM_NAME)?;
            Ok(uniforms
                .into_iter()
                .map(|u| poisson_inv(u, fit.parameters[0]))
                .collect())
        }
    }
}

async fn parse_shape_args(rest: &[Value]) -> BuiltinResult<Vec<usize>> {
    if rest.is_empty() {
        return Ok(vec![1, 1]);
    }
    let mut dims = Vec::new();
    for arg in rest {
        match extract_dims(arg, RANDOM_NAME).await {
            Ok(Some(values)) => dims.extend(values),
            Ok(None) => return Err(invalid("random: invalid size argument")),
            Err(err) => return Err(invalid(err)),
        }
    }
    Ok(normalize_dims(dims))
}

fn normalize_dims(dims: Vec<usize>) -> Vec<usize> {
    if dims.is_empty() {
        vec![0, 0]
    } else if dims.len() == 1 {
        vec![dims[0], dims[0]]
    } else {
        dims
    }
}

fn finish_for(builtin: &'static str, shape: Vec<usize>, data: Vec<f64>) -> BuiltinResult<Value> {
    if shape.iter().copied().product::<usize>() == 1 {
        return Ok(Value::Num(data.first().copied().unwrap_or(0.0)));
    }
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| internal_for(builtin, format!("{builtin}: {err}")))
}

fn two(values: &[f64]) -> [f64; 2] {
    [values[0], values[1]]
}

fn invert_positive(p: f64, initial_hi: f64, cdf: impl Fn(f64) -> f64) -> f64 {
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let mut lo = 0.0;
    let mut hi = initial_hi.max(1.0);
    let mut iter = 0;
    while cdf(hi) < p {
        hi *= 2.0;
        iter += 1;
        if !hi.is_finite() || iter > 2048 {
            return f64::INFINITY;
        }
    }
    for _ in 0..160 {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

fn poisson_inv(p: f64, lambda: f64) -> f64 {
    if p.is_nan() || lambda.is_nan() || lambda < 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if lambda == 0.0 || p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let mut k = 0.0;
    loop {
        if distribution_math::regularized_gamma_q(k + 1.0, lambda) >= p {
            return k;
        }
        k += 1.0;
        if k > lambda + 20.0 * lambda.sqrt().max(1.0) + 1000.0 {
            return k;
        }
    }
}

fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 8.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + inv + 0.5 * inv2 + inv2 * inv / 6.0 - inv2 * inv2 * inv / 30.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn vec_tensor(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![values.len(), 1]).unwrap())
    }

    fn int_vec_tensor(storage: IntegerStorage, len: usize) -> Value {
        Value::Tensor(Tensor::new_integer(storage, vec![len, 1]).unwrap())
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        Value::Tensor(tensor)
    }

    fn mirrorless_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        Value::Tensor(tensor)
    }

    fn all_cdf_integer_storages(value: i8) -> Vec<IntegerStorage> {
        vec![
            IntegerStorage::I8(vec![value]),
            IntegerStorage::I16(vec![i16::from(value)]),
            IntegerStorage::I32(vec![i32::from(value)]),
            IntegerStorage::I64(vec![i64::from(value)]),
            IntegerStorage::U8(vec![value as u8]),
            IntegerStorage::U16(vec![value as u16]),
            IntegerStorage::U32(vec![value as u32]),
            IntegerStorage::U64(vec![value as u64]),
        ]
    }

    fn all_fitdist_integer_storages() -> Vec<IntegerStorage> {
        vec![
            IntegerStorage::I8(vec![1, 2, 3]),
            IntegerStorage::I16(vec![1, 2, 3]),
            IntegerStorage::I32(vec![1, 2, 3]),
            IntegerStorage::I64(vec![1, 2, 3]),
            IntegerStorage::U8(vec![1, 2, 3]),
            IntegerStorage::U16(vec![1, 2, 3]),
            IntegerStorage::U32(vec![1, 2, 3]),
            IntegerStorage::U64(vec![1, 2, 3]),
        ]
    }

    #[test]
    fn fitdist_broadcast_indexing_appends_trailing_singletons() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        assert_eq!(
            broadcast_tensor_to(PDF_NAME, &tensor, &[2, 1, 3]).unwrap(),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
    }

    fn object(value: Value) -> ObjectInstance {
        match value {
            Value::Object(object) => object,
            other => panic!("expected object, got {other:?}"),
        }
    }

    #[test]
    fn fitdist_normal_fits_and_evaluates_object_methods() {
        let pd = block_on(fitdist_builtin(
            vec_tensor(&[1.0, 2.0, 3.0]),
            Value::String("Normal".into()),
            Vec::new(),
        ))
        .unwrap();
        let object = object(pd.clone());
        assert_eq!(
            object.properties.get("DistributionName"),
            Some(&Value::String("Normal".into()))
        );
        let values = numeric_vector_property(&object, "ParameterValues").unwrap();
        assert!((values[0] - 2.0).abs() < 1.0e-12);
        assert!((values[1] - (2.0_f64 / 3.0).sqrt()).abs() < 1.0e-12);

        let density = block_on(pdf_builtin(pd.clone(), Value::Num(2.0), Vec::new())).unwrap();
        let Value::Num(density) = density else {
            panic!("expected scalar pdf");
        };
        assert!(density > 0.0);

        let p = block_on(cdf_builtin(pd.clone(), Value::Num(2.0), Vec::new())).unwrap();
        assert_eq!(p, Value::Num(0.5));

        let upper = block_on(cdf_builtin(
            pd.clone(),
            Value::Num(10.0),
            vec![Value::String("upper".into())],
        ))
        .unwrap();
        assert!(matches!(upper, Value::Num(value) if value > 0.0 && value < 1.0e-20));

        let x = block_on(icdf_probability_distribution(pd, Value::Num(0.5))).unwrap();
        assert_eq!(x, Value::Num(2.0));
    }

    #[test]
    fn fitdist_accepts_typed_integer_sample_and_eval_points() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let pd = block_on(fitdist_builtin(
            int_vec_tensor(IntegerStorage::I16(vec![1, 2, 3]), 3),
            Value::String("Normal".into()),
            Vec::new(),
        ))
        .unwrap();
        let object = object(pd.clone());
        let values = numeric_vector_property(&object, "ParameterValues").unwrap();
        assert!((values[0] - 2.0).abs() < 1.0e-12);

        let density = block_on(pdf_builtin(
            pd,
            mirrorless_int_tensor(IntegerStorage::U16(vec![2, 3]), vec![2, 1]),
            Vec::new(),
        ))
        .unwrap();
        match density {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert!(tensor
                    .materialize_f64()
                    .iter()
                    .all(|value| value.is_finite()));
            }
            other => panic!("expected tensor density, got {other:?}"),
        }
    }

    #[test]
    fn fitdist_integer_data_and_frequency_are_separate_gated_extensions() {
        for storage in all_fitdist_integer_storages() {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            block_on(fitdist_builtin(
                mirrorless_int_tensor(storage.clone(), vec![3, 1]),
                Value::String("Normal".into()),
                Vec::new(),
            ))
            .expect("integer observations");
            block_on(fitdist_builtin(
                vec_tensor(&[1.0, 2.0, 3.0]),
                Value::String("Normal".into()),
                vec![
                    Value::from("Frequency"),
                    mirrorless_int_tensor(storage, vec![3, 1]),
                ],
            ))
            .expect("integer Frequency");
        }

        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let data_error = block_on(fitdist_builtin(
            mirrorless_int_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]),
            Value::String("Normal".into()),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(
            data_error.identifier(),
            Some("RunMat:compatibility:FitdistIntegerDataExtension")
        );
        let frequency_error = block_on(fitdist_builtin(
            vec_tensor(&[1.0, 2.0, 3.0]),
            Value::String("Normal".into()),
            vec![
                Value::from("Frequency"),
                mirrorless_int_tensor(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1]),
            ],
        ))
        .unwrap_err();
        assert_eq!(
            frequency_error.identifier(),
            Some("RunMat:compatibility:FitdistIntegerFrequencyExtension")
        );
    }

    #[test]
    fn fitdist_rejects_inexact_integer_boundaries_and_fractional_frequency() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let exact_wide = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 54]), vec![1, 1])
            .expect("exact wide integer");
        ensure_exact_fitdist_integer_boundary(&exact_wide, "test")
            .expect("wide powers of two remain exactly representable");
        let wide = block_on(fitdist_builtin(
            mirrorless_int_tensor(
                IntegerStorage::U64(vec![1, 2, (1_u64 << 53) + 1]),
                vec![3, 1],
            ),
            Value::String("Normal".into()),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(wide.message().contains("exactly representable as double"));

        for invalid in [1.5, f64::NAN, f64::INFINITY, -1.0] {
            let error = block_on(fitdist_builtin(
                vec_tensor(&[1.0, 2.0]),
                Value::String("Normal".into()),
                vec![Value::from("Frequency"), vec_tensor(&[0.0, invalid])],
            ))
            .unwrap_err();
            assert!(error.message().contains("integer counts"));
        }
    }

    #[test]
    fn fitdist_strict_mode_gates_resident_fallback_before_gather() {
        use crate::builtins::common::test_support;

        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1])
                .expect("integer observations");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

            let floating = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("observations");
            let floating_handle =
                gpu_helpers::upload_tensor(provider, &floating).expect("floating upload");
            let resident_error = block_on(fitdist_builtin(
                Value::GpuTensor(floating_handle.clone()),
                Value::String("Normal".into()),
                Vec::new(),
            ))
            .unwrap_err();
            assert_eq!(
                resident_error.identifier(),
                Some("RunMat:compatibility:FitdistResidentFallbackExtension")
            );
            assert!(runmat_accelerate_api::provider_for_handle(&floating_handle)
                .is_some_and(|owner| std::ptr::eq(owner, provider)));
            let resident_option_error = block_on(fitdist_builtin(
                vec_tensor(&[1.0, 2.0, 3.0]),
                Value::String("Normal".into()),
                vec![
                    Value::from("Frequency"),
                    Value::GpuTensor(floating_handle.clone()),
                ],
            ))
            .unwrap_err();
            assert_eq!(
                resident_option_error.identifier(),
                Some("RunMat:compatibility:FitdistResidentFallbackExtension")
            );

            let error = block_on(fitdist_builtin(
                Value::GpuTensor(handle.clone()),
                Value::String("Normal".into()),
                Vec::new(),
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:FitdistIntegerDataExtension")
            );
            assert!(runmat_accelerate_api::provider_for_handle(&handle)
                .is_some_and(|owner| std::ptr::eq(owner, provider)));

            let frequency = Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1])
                .expect("integer Frequency");
            let frequency_handle =
                gpu_helpers::upload_tensor(provider, &frequency).expect("integer upload");
            let error = block_on(fitdist_builtin(
                vec_tensor(&[1.0, 2.0, 3.0]),
                Value::String("Normal".into()),
                vec![
                    Value::from("Frequency"),
                    Value::GpuTensor(frequency_handle.clone()),
                ],
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:FitdistIntegerFrequencyExtension")
            );
            assert!(
                runmat_accelerate_api::provider_for_handle(&frequency_handle)
                    .is_some_and(|owner| std::ptr::eq(owner, provider))
            );
        });
    }

    #[test]
    fn fitdist_random_shape_parser_ignores_all_typed_mirrors() {
        let storages = [
            IntegerStorage::I8(vec![2, 3]),
            IntegerStorage::I16(vec![2, 3]),
            IntegerStorage::I32(vec![2, 3]),
            IntegerStorage::I64(vec![2, 3]),
            IntegerStorage::U8(vec![2, 3]),
            IntegerStorage::U16(vec![2, 3]),
            IntegerStorage::U32(vec![2, 3]),
            IntegerStorage::U64(vec![2, 3]),
        ];

        for storage in storages {
            assert_eq!(
                block_on(parse_shape_args(&[mirrorless_int_tensor(
                    storage,
                    vec![2, 1]
                )]))
                .unwrap(),
                vec![2, 3]
            );
        }
    }

    #[test]
    fn fitdist_reads_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let pd = block_on(fitdist_builtin(
            poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]),
            Value::String("Exponential".into()),
            vec![
                Value::String("Frequency".into()),
                mirrorless_int_tensor(IntegerStorage::U8(vec![1, 3, 2]), vec![3, 1]),
            ],
        ))
        .unwrap();
        let object = object(pd.clone());
        let values = numeric_vector_property(&object, "ParameterValues").unwrap();
        assert!((values[0] - (13.0 / 6.0)).abs() < 1.0e-12);

        let mut typed_object = object;
        typed_object.properties.insert(
            "ParameterValues".to_string(),
            poisoned_int_tensor(IntegerStorage::U16(vec![7]), vec![1, 1]),
        );
        assert_eq!(
            numeric_vector_property(&typed_object, "ParameterValues").unwrap(),
            vec![7.0]
        );
    }

    #[test]
    fn fitdist_sample_length_uses_typed_integer_storage_not_mirror() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let pd = block_on(fitdist_builtin(
            mirrorless_int_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]),
            Value::String("Normal".into()),
            Vec::new(),
        ))
        .unwrap();
        let object = object(pd);
        let values = numeric_vector_property(&object, "ParameterValues").unwrap();
        assert!((values[0] - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn fitdist_frequency_and_range_validation() {
        let pd = object(
            block_on(fitdist_builtin(
                vec_tensor(&[1.0, 2.0]),
                Value::String("Exponential".into()),
                vec![
                    Value::String("Frequency".into()),
                    Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
                ],
            ))
            .unwrap(),
        );
        let values = numeric_vector_property(&pd, "ParameterValues").unwrap();
        assert!((values[0] - 1.75).abs() < 1.0e-12);

        let err = block_on(fitdist_builtin(
            vec_tensor(&[-1.0, 2.0]),
            Value::String("Gamma".into()),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(err.message.contains("outside the supported range"));
    }

    #[test]
    fn fitdist_gamma_weibull_and_poisson_smoke() {
        let gamma = block_on(fitdist_builtin(
            vec_tensor(&[1.0, 2.0, 4.0, 8.0]),
            Value::String("Gamma".into()),
            Vec::new(),
        ))
        .unwrap();
        assert!(matches!(
            block_on(cdf_builtin(gamma, Value::Num(2.0), Vec::new())).unwrap(),
            Value::Num(value) if value.is_finite()
        ));

        let weibull = block_on(fitdist_builtin(
            vec_tensor(&[1.0, 2.0, 3.0, 5.0, 8.0]),
            Value::String("Weibull".into()),
            Vec::new(),
        ))
        .unwrap();
        assert!(matches!(
            block_on(pdf_builtin(weibull, Value::Num(2.0), Vec::new())).unwrap(),
            Value::Num(value) if value.is_finite() && value >= 0.0
        ));

        let poisson = block_on(fitdist_builtin(
            vec_tensor(&[0.0, 1.0, 1.0, 2.0, 3.0]),
            Value::String("Poisson".into()),
            Vec::new(),
        ))
        .unwrap();
        let samples = block_on(random_builtin(poisson, vec![Value::Num(2.0)])).unwrap();
        match samples {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![2, 2]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn generic_pdf_cdf_random_name_overloads_execute() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let density = block_on(pdf_builtin(
            Value::String("Normal".into()),
            mirrorless_int_tensor(IntegerStorage::I16(vec![0, 1]), vec![2, 1]),
            vec![
                mirrorless_int_tensor(IntegerStorage::I16(vec![0]), vec![1, 1]),
                mirrorless_int_tensor(IntegerStorage::I16(vec![1]), vec![1, 1]),
            ],
        ))
        .unwrap();
        let Value::Tensor(density) = density else {
            panic!("expected vector density");
        };
        assert_eq!(density.shape, vec![2, 1]);
        assert!(
            (density.materialize_f64()[0] - distribution_math::standard_normal_pdf(0.0)).abs()
                < 1.0e-12
        );

        let probability = block_on(cdf_builtin(
            Value::String("Poisson".into()),
            Value::Num(2.0),
            vec![Value::Num(3.0)],
        ))
        .unwrap();
        let Value::Num(probability) = probability else {
            panic!("expected scalar probability");
        };
        assert!(probability > 0.0 && probability < 1.0);

        let samples = block_on(random_builtin(
            Value::String("Weibull".into()),
            vec![
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(2.0),
                Value::Num(3.0),
            ],
        ))
        .unwrap();
        match samples {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![2, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn random_restores_parameter_residency_but_not_size_control_residency() {
        use crate::builtins::common::test_support;

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let parameter = Tensor::new_integer(IntegerStorage::I16(vec![0]), vec![1, 1])
                .expect("integer parameter");
            let parameter = gpu_helpers::upload_tensor(provider, &parameter).expect("upload");
            let parameter =
                parameter.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let output = block_on(random_builtin(
                Value::String("Normal".into()),
                vec![
                    Value::GpuTensor(parameter),
                    Value::Num(1.0),
                    Value::Num(2.0),
                ],
            ))
            .expect("resident random output");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("expected resident random output");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
            let gathered = test_support::gather(output).expect("gather random output");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert!(gathered
                .materialize_f64()
                .iter()
                .all(|value| value.is_finite()));

            let size =
                Tensor::new_integer(IntegerStorage::U8(vec![2]), vec![1, 1]).expect("integer size");
            let size = gpu_helpers::upload_tensor(provider, &size).expect("upload size");
            let size = size.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let output = block_on(random_builtin(
                Value::String("Normal".into()),
                vec![Value::Num(0.0), Value::Num(1.0), Value::GpuTensor(size)],
            ))
            .expect("resident size control");
            assert!(matches!(output, Value::Tensor(_)));
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn random_wgpu_parameter_residency_enforces_double_for_all_integer_classes() {
        use crate::builtins::common::test_support;

        let _accel_guard = test_support::accel_test_lock();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in all_cdf_integer_storages(0) {
            let parameter = Tensor::new_integer(storage, vec![1, 1]).expect("integer parameter");
            let handle = gpu_helpers::upload_tensor(provider, &parameter).expect("upload");
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let result = block_on(random_builtin(
                Value::String("Normal".into()),
                vec![Value::GpuTensor(handle), Value::Num(1.0), Value::Num(2.0)],
            ));
            if provider.precision() == ProviderPrecision::F64 {
                let output = result.expect("resident random output");
                let Value::GpuTensor(handle) = &output else {
                    panic!("expected resident random output");
                };
                assert!(runmat_accelerate_api::handle_is_explicit(handle));
                let gathered = test_support::gather(output).expect("gather random output");
                assert_eq!(gathered.shape, vec![2, 2]);
                assert!(gathered
                    .materialize_f64()
                    .iter()
                    .all(|value| value.is_finite()));
            } else {
                let error = result.expect_err("f32 owner cannot preserve double output");
                assert!(error
                    .message()
                    .contains("cannot preserve explicit gpuArray"));
            }
        }
    }

    #[test]
    fn pdf_broadcasts_parameters_preserves_single_and_restores_residency() {
        use crate::builtins::common::test_support;

        let density = block_on(pdf_builtin(
            Value::String("Normal".into()),
            Value::Tensor(Tensor::from_f32(vec![0.0, 2.0], vec![2, 1]).unwrap()),
            vec![
                Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
                Value::Num(1.0),
            ],
        ))
        .expect("broadcast single pdf");
        let Value::Tensor(density) = density else {
            panic!("expected single tensor");
        };
        assert_eq!(density.numeric_dtype(), NumericDType::F32);
        assert_eq!(density.shape, vec![2, 1]);

        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let integer = Tensor::new_integer(IntegerStorage::I16(vec![0, 1]), vec![2, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &integer).expect("integer upload");
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let density = block_on(pdf_builtin(
                Value::String("Normal".into()),
                Value::GpuTensor(handle),
                vec![Value::Num(0.0), Value::Num(1.0)],
            ))
            .expect("resident integer pdf");
            assert!(matches!(density, Value::GpuTensor(_)));
            let gathered = test_support::gather(density).expect("gather pdf");
            assert_eq!(gathered.shape, vec![2, 1]);
        });
    }

    #[test]
    fn cdf_classifies_all_integer_input_positions_as_runmat_extensions() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in all_cdf_integer_storages(1) {
            let x = block_on(cdf_builtin(
                Value::String("Normal".into()),
                mirrorless_int_tensor(storage.clone(), vec![1, 1]),
                vec![Value::Num(0.0), Value::Num(1.0)],
            ))
            .expect("integer x");
            assert!(
                matches!(x, Value::Num(value) if (value - 0.841_344_746_068_543).abs() < 1.0e-12)
            );

            let parameter = block_on(cdf_builtin(
                Value::String("Poisson".into()),
                Value::Num(0.0),
                vec![mirrorless_int_tensor(storage, vec![1, 1])],
            ))
            .expect("integer parameter");
            assert!(
                matches!(parameter, Value::Num(value) if (value - (-1.0_f64).exp()).abs() < 1.0e-12)
            );
        }

        let pd = block_on(fitdist_builtin(
            vec_tensor(&[0.0, 1.0, 2.0]),
            Value::String("Normal".into()),
            Vec::new(),
        ))
        .expect("fitted normal");
        let object_x = block_on(cdf_builtin(
            pd,
            mirrorless_int_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
            Vec::new(),
        ))
        .expect("integer object evaluation point");
        assert_eq!(object_x, Value::Num(0.5));
    }

    #[test]
    fn cdf_compatibility_mode_rejects_integer_and_logical_extensions() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer_x = block_on(cdf_builtin(
            Value::String("Normal".into()),
            mirrorless_int_tensor(IntegerStorage::I8(vec![1]), vec![1, 1]),
            vec![Value::Num(0.0), Value::Num(1.0)],
        ))
        .unwrap_err();
        assert_eq!(
            integer_x.identifier(),
            Some("RunMat:compatibility:CdfIntegerXExtension")
        );
        let pd = block_on(fitdist_builtin(
            vec_tensor(&[0.0, 1.0, 2.0]),
            Value::String("Normal".into()),
            Vec::new(),
        ))
        .expect("fitted normal");
        let object_integer_x = block_on(cdf_builtin(
            pd,
            mirrorless_int_tensor(IntegerStorage::I16(vec![1]), vec![1, 1]),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(
            object_integer_x.identifier(),
            Some("RunMat:compatibility:CdfIntegerXExtension")
        );
        let integer_parameter = block_on(cdf_builtin(
            Value::String("Poisson".into()),
            Value::Num(0.0),
            vec![mirrorless_int_tensor(
                IntegerStorage::U16(vec![1]),
                vec![1, 1],
            )],
        ))
        .unwrap_err();
        assert_eq!(
            integer_parameter.identifier(),
            Some("RunMat:compatibility:CdfIntegerParametersExtension")
        );
        let logical = block_on(cdf_builtin(
            Value::String("Normal".into()),
            Value::Bool(true),
            vec![Value::Num(0.0), Value::Num(1.0)],
        ))
        .unwrap_err();
        assert_eq!(
            logical.identifier(),
            Some("RunMat:compatibility:CdfLogicalInputExtension")
        );
    }

    #[test]
    fn cdf_broadcasts_parameters_supports_upper_and_preserves_single() {
        let result = block_on(cdf_builtin(
            Value::String("Normal".into()),
            Value::Tensor(Tensor::from_f32(vec![0.0, 2.0], vec![2, 1]).unwrap()),
            vec![
                Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
                Value::Num(1.0),
                Value::String("upper".into()),
            ],
        ))
        .expect("broadcast upper cdf");
        let Value::Tensor(result) = result else {
            panic!("single output must retain its class");
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
        assert_eq!(result.shape, vec![2, 1]);
        let values = result.materialize_f64();
        assert!((values[0] - 0.5).abs() < f64::from(f32::EPSILON));
        assert!((values[1] - 0.158_655_253_931_457).abs() < f64::from(f32::EPSILON));

        let tail = block_on(cdf_builtin(
            Value::String("Exponential".into()),
            Value::Num(1000.0),
            vec![Value::Num(1.0), Value::String("upper".into())],
        ))
        .expect("upper tail");
        assert_eq!(tail, Value::Num(0.0));
    }

    #[test]
    fn cdf_rejects_inexact_wide_integers() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = block_on(cdf_builtin(
            Value::String("Normal".into()),
            mirrorless_int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]),
            vec![Value::Num(0.0), Value::Num(1.0)],
        ))
        .unwrap_err();
        assert!(error.message().contains("exactly representable as double"));
    }

    #[test]
    fn cdf_gpu_fallback_preserves_residency_precision_and_guard_order() {
        use crate::builtins::common::test_support;

        test_support::with_test_provider(|provider| {
            let single = Tensor::from_f32(vec![1.0], vec![1, 1]).expect("single input");
            let handle = gpu_helpers::upload_tensor(provider, &single).expect("single upload");
            runmat_accelerate_api::set_handle_precision(&handle, ProviderPrecision::F32);
            let result = block_on(cdf_builtin(
                Value::String("Normal".into()),
                Value::GpuTensor(handle),
                vec![Value::Num(0.0), Value::Num(1.0)],
            ))
            .expect("resident cdf");
            let Value::GpuTensor(result_handle) = &result else {
                panic!("expected resident output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(result_handle),
                Some(ProviderPrecision::F32)
            );
            let gathered = test_support::gather(result).expect("gather result");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);

            let integer =
                Tensor::new_integer(IntegerStorage::I16(vec![1]), vec![1, 1]).expect("integer x");
            let handle = gpu_helpers::upload_tensor(provider, &integer).expect("integer upload");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(cdf_builtin(
                Value::String("Normal".into()),
                Value::GpuTensor(handle),
                vec![Value::Num(0.0), Value::Num(1.0)],
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:CdfIntegerXExtension")
            );
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn pdf_wgpu_fallback_preserves_residency_for_all_integer_classes() {
        use crate::builtins::common::test_support;

        let _accel_guard = test_support::accel_test_lock();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in all_cdf_integer_storages(1) {
            let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer x");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let result = block_on(pdf_builtin(
                Value::String("Normal".into()),
                Value::GpuTensor(handle),
                vec![Value::Num(0.0), Value::Num(1.0)],
            ))
            .expect("resident integer pdf");
            let Value::GpuTensor(output) = &result else {
                panic!("expected resident pdf output");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(output));
            let gathered = test_support::gather(result).expect("gather result");
            assert!((gathered.materialize_f64()[0] - 0.241_970_724_519_143_37).abs() < 1.0e-12);
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn cdf_wgpu_fallback_preserves_residency_for_all_integer_classes() {
        use crate::builtins::common::test_support;

        let _accel_guard = test_support::accel_test_lock();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in all_cdf_integer_storages(1) {
            let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer x");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
            let result = block_on(cdf_builtin(
                Value::String("Normal".into()),
                Value::GpuTensor(handle),
                vec![Value::Num(0.0), Value::Num(1.0)],
            ))
            .expect("resident integer cdf");
            assert!(matches!(result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).expect("gather result");
            assert!((gathered.materialize_f64()[0] - 0.841_344_746_068_543).abs() < 1.0e-12);
        }
    }
}
