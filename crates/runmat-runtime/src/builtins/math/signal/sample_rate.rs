//! MATLAB-compatible sample-rate conversion builtins.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, IntegerStorage, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::signal::type_resolvers::{
    downsample_type, resample_type, upsample_type,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const UPSAMPLE_NAME: &str = "upsample";
const DOWNSAMPLE_NAME: &str = "downsample";
const RESAMPLE_NAME: &str = "resample";
const DEFAULT_RESAMPLE_N: usize = 10;
const DEFAULT_RESAMPLE_BETA: f64 = 5.0;

const DOWNSAMPLE_INTEGER_FACTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "downsample-integer-factor",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "downsample with a typed-integer factor is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DownsampleIntegerFactorExtension"),
    };
const DOWNSAMPLE_INTEGER_PHASE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "downsample-integer-phase",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "downsample with a typed-integer phase is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DownsampleIntegerPhaseExtension"),
};
const DOWNSAMPLE_ND_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "downsample-nd-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "downsample with an input having more than two dimensions is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DownsampleNdInputExtension"),
};
const DOWNSAMPLE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    DOWNSAMPLE_INTEGER_FACTOR_EXTENSION,
    DOWNSAMPLE_INTEGER_PHASE_EXTENSION,
    DOWNSAMPLE_ND_INPUT_EXTENSION,
];
const DOWNSAMPLE_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public X argument is a vector or matrix without a datatype restriction. Downsampling is structural selection, so authoritative integer elements and their class are preserved exactly.",
    }];
const DOWNSAMPLE_INTEGER_FACTOR_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "N",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target lists single and double for N; RunMat mode additionally parses typed integers exactly as host sizes.",
    }];
const DOWNSAMPLE_INTEGER_PHASE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "PHASE",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target lists single and double for PHASE; RunMat mode additionally parses typed integers exactly before validating 0 <= PHASE < N.",
    }];
pub const DOWNSAMPLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = downsample(integer_X, N, PHASE?)",
        inputs: &DOWNSAMPLE_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Host storage is selected without conversion; resident integer storage uses the owner-resolved linear-gather hook and is accepted only with matching owner, device, shape, storage, and integer class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = downsample(X, integer_N, PHASE?)",
        inputs: &DOWNSAMPLE_INTEGER_FACTOR_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The factor is extension-gated before X or any provider is accessed, then converted exactly to a host size.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = downsample(X, N, integer_PHASE)",
        inputs: &DOWNSAMPLE_INTEGER_PHASE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The phase is extension-gated before X or any provider is accessed, parsed exactly, and range-checked against N.",
    },
];

const SAMPLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample-rate-adjusted signal.",
}];

const SAMPLE_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal array.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer sample-rate factor.",
    },
];

const SAMPLE_INPUTS_PHASE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal array.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer sample-rate factor.",
    },
    BuiltinParamDescriptor {
        name: "PHASE",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Integer phase offset in the range 0 <= PHASE < N.",
    },
];

const UPSAMPLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = upsample(X, N)",
        inputs: &SAMPLE_INPUTS_CORE,
        outputs: &SAMPLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = upsample(X, N, PHASE)",
        inputs: &SAMPLE_INPUTS_PHASE,
        outputs: &SAMPLE_OUTPUT,
    },
];

const DOWNSAMPLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = downsample(X, N)",
        inputs: &SAMPLE_INPUTS_CORE,
        outputs: &SAMPLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = downsample(X, N, PHASE)",
        inputs: &SAMPLE_INPUTS_PHASE,
        outputs: &SAMPLE_OUTPUT,
    },
];

const RESAMPLE_INPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input signal array.",
};

const RESAMPLE_INPUT_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "P",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive integer upsampling factor.",
};

const RESAMPLE_INPUT_Q: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive integer downsampling factor.",
};

const RESAMPLE_INPUT_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("10"),
    description: "Half-length filter order factor.",
};

const RESAMPLE_INPUT_BETA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "BETA",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("5"),
    description: "Kaiser window shape parameter.",
};

const RESAMPLE_INPUT_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "FIR antialiasing filter coefficients.",
};

const RESAMPLE_INPUTS_CORE: [BuiltinParamDescriptor; 3] =
    [RESAMPLE_INPUT_X, RESAMPLE_INPUT_P, RESAMPLE_INPUT_Q];

const RESAMPLE_INPUTS_N: [BuiltinParamDescriptor; 4] = [
    RESAMPLE_INPUT_X,
    RESAMPLE_INPUT_P,
    RESAMPLE_INPUT_Q,
    RESAMPLE_INPUT_N,
];

const RESAMPLE_INPUTS_N_BETA: [BuiltinParamDescriptor; 5] = [
    RESAMPLE_INPUT_X,
    RESAMPLE_INPUT_P,
    RESAMPLE_INPUT_Q,
    RESAMPLE_INPUT_N,
    RESAMPLE_INPUT_BETA,
];

const RESAMPLE_INPUTS_FILTER: [BuiltinParamDescriptor; 4] = [
    RESAMPLE_INPUT_X,
    RESAMPLE_INPUT_P,
    RESAMPLE_INPUT_Q,
    RESAMPLE_INPUT_B,
];

const RESAMPLE_INPUTS_FILTER_VARIADIC: [BuiltinParamDescriptor; 4] = [
    RESAMPLE_INPUT_X,
    RESAMPLE_INPUT_P,
    RESAMPLE_INPUT_Q,
    BuiltinParamDescriptor {
        arity: BuiltinParamArity::Variadic,
        ..RESAMPLE_INPUT_B
    },
];

const RESAMPLE_OUTPUT_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Resampled signal.",
};

const RESAMPLE_OUTPUT_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "FIR antialiasing filter coefficients used for resampling.",
};

const RESAMPLE_OUTPUTS_Y: [BuiltinParamDescriptor; 1] = [RESAMPLE_OUTPUT_Y];
const RESAMPLE_OUTPUTS_Y_B: [BuiltinParamDescriptor; 2] = [RESAMPLE_OUTPUT_Y, RESAMPLE_OUTPUT_B];

const RESAMPLE_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "Y = resample(X, P, Q)",
        inputs: &RESAMPLE_INPUTS_CORE,
        outputs: &RESAMPLE_OUTPUTS_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = resample(X, P, Q, N)",
        inputs: &RESAMPLE_INPUTS_N,
        outputs: &RESAMPLE_OUTPUTS_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = resample(X, P, Q, N, BETA)",
        inputs: &RESAMPLE_INPUTS_N_BETA,
        outputs: &RESAMPLE_OUTPUTS_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = resample(X, P, Q, B)",
        inputs: &RESAMPLE_INPUTS_FILTER,
        outputs: &RESAMPLE_OUTPUTS_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[Y, B] = resample(X, P, Q, ___)",
        inputs: &RESAMPLE_INPUTS_N_BETA,
        outputs: &RESAMPLE_OUTPUTS_Y_B,
    },
    BuiltinSignatureDescriptor {
        label: "[Y, B] = resample(X, P, Q, B, ___)",
        inputs: &RESAMPLE_INPUTS_FILTER_VARIADIC,
        outputs: &RESAMPLE_OUTPUTS_Y_B,
    },
];

const SAMPLE_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.ARG_COUNT",
    identifier: Some("RunMat:sampleRate:ArgCount"),
    when: "Required arguments are missing or more than three inputs are provided.",
    message: "sample-rate builtin: expected X, N, and optional PHASE",
};

const SAMPLE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.INVALID_INPUT",
    identifier: Some("RunMat:sampleRate:InvalidInput"),
    when: "X is not numeric, logical, complex, or a gpuArray containing numeric values.",
    message: "sample-rate builtin: X must be numeric or logical",
};

const SAMPLE_ERROR_INVALID_FACTOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.INVALID_FACTOR",
    identifier: Some("RunMat:sampleRate:InvalidFactor"),
    when: "N is not a finite positive integer scalar.",
    message: "sample-rate builtin: N must be a positive integer scalar",
};

const SAMPLE_ERROR_INVALID_PHASE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.INVALID_PHASE",
    identifier: Some("RunMat:sampleRate:InvalidPhase"),
    when: "PHASE is not an integer scalar in the range 0 <= PHASE < N.",
    message: "sample-rate builtin: PHASE must be an integer scalar with 0 <= PHASE < N",
};

const SAMPLE_ERROR_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.GATHER_FAILED",
    identifier: Some("RunMat:sampleRate:GatherFailed"),
    when: "A gpuArray input cannot be gathered for host sample-rate conversion.",
    message: "sample-rate builtin: failed to gather GPU input",
};

const SAMPLE_ERROR_SIZE_OVERFLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.SIZE_OVERFLOW",
    identifier: Some("RunMat:sampleRate:SizeOverflow"),
    when: "The requested output shape overflows host allocation limits.",
    message: "sample-rate builtin: requested output is too large",
};

const SAMPLE_ERROR_BUILD_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.BUILD_OUTPUT",
    identifier: Some("RunMat:sampleRate:BuildOutput"),
    when: "Output tensor construction fails.",
    message: "sample-rate builtin: failed to build output",
};

const SAMPLE_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.INVALID_OPTION",
    identifier: Some("RunMat:sampleRate:InvalidOption"),
    when: "A sample-rate conversion option, filter vector, or dimension selector is invalid.",
    message: "sample-rate builtin: invalid option",
};

const SAMPLE_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.OUTPUT_COUNT",
    identifier: Some("RunMat:sampleRate:OutputCount"),
    when: "More outputs are requested than the builtin supports.",
    message: "sample-rate builtin: too many output arguments",
};

const BASIC_SAMPLE_ERRORS: [BuiltinErrorDescriptor; 7] = [
    SAMPLE_ERROR_ARG_COUNT,
    SAMPLE_ERROR_INVALID_INPUT,
    SAMPLE_ERROR_INVALID_FACTOR,
    SAMPLE_ERROR_INVALID_PHASE,
    SAMPLE_ERROR_GATHER_FAILED,
    SAMPLE_ERROR_SIZE_OVERFLOW,
    SAMPLE_ERROR_BUILD_OUTPUT,
];

const RESAMPLE_ERRORS: [BuiltinErrorDescriptor; 9] = [
    SAMPLE_ERROR_ARG_COUNT,
    SAMPLE_ERROR_INVALID_INPUT,
    SAMPLE_ERROR_INVALID_FACTOR,
    SAMPLE_ERROR_INVALID_PHASE,
    SAMPLE_ERROR_GATHER_FAILED,
    SAMPLE_ERROR_SIZE_OVERFLOW,
    SAMPLE_ERROR_BUILD_OUTPUT,
    SAMPLE_ERROR_INVALID_OPTION,
    SAMPLE_ERROR_OUTPUT_COUNT,
];

pub const UPSAMPLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UPSAMPLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BASIC_SAMPLE_ERRORS,
};

pub const DOWNSAMPLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DOWNSAMPLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BASIC_SAMPLE_ERRORS,
};

pub const RESAMPLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RESAMPLE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RESAMPLE_ERRORS,
};

const RESAMPLE_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "resample-integer-options",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "resample accepts typed-integer rate, filter-design, filter, and dimension arguments as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ResampleIntegerOptionsExtension"),
};
const RESAMPLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [RESAMPLE_INTEGER_OPTION_EXTENSION];
const RESAMPLE_REJECTED_INTEGER_DATA: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double signal input; native integer signals reject before host or provider execution.",
    }];
const RESAMPLE_INTEGER_STRUCTURAL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "P/Q/N/Dimension",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target lists single and double controls; RunMat mode additionally reads typed integer rate and dimension controls exactly as structural values.",
    }];
const RESAMPLE_INTEGER_FLOAT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "BETA/B",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed Kaiser beta or FIR coefficients are a RunMat extension and enter filtering only after exact binary64 representability is proved.",
    }];
pub const RESAMPLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "resample(integer_X, P, Q, ...)",
        inputs: &RESAMPLE_REJECTED_INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer signal storage is outside the compatibility surface and rejects from authoritative host or resident dtype metadata before filtering.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = resample(X, integer_P, integer_Q, integer_N, Dimension=integer_dim)",
        inputs: &RESAMPLE_INTEGER_STRUCTURAL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Rate and dimension controls remain exact through validation; checked arithmetic owns output and filter geometry.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = resample(X, P, Q, integer_BETA_or_B)",
        inputs: &RESAMPLE_INTEGER_FLOAT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Filter-design or coefficient integers cross one named checked binary64 FIR boundary; X retains its documented floating output class and residency behavior.",
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleOp {
    Up,
    Down,
}

enum SampleInput {
    Real {
        storage: NumericStorage,
        shape: Vec<usize>,
    },
    Complex {
        data: Vec<(f64, f64)>,
        shape: Vec<usize>,
    },
}

impl SampleInput {
    async fn from_value(value: Value, builtin: &'static str) -> BuiltinResult<Self> {
        match value {
            Value::GpuTensor(handle) => {
                let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                    .await
                    .map_err(|flow| {
                        let detail = flow.message().to_owned();
                        sample_error_with_source(
                            builtin,
                            &SAMPLE_ERROR_GATHER_FAILED,
                            detail,
                            map_control_flow_with_builtin(flow, builtin),
                        )
                    })?;
                match gathered {
                    Value::Tensor(tensor) => Self::from_tensor(tensor, builtin),
                    Value::LogicalArray(logical) => {
                        let tensor = tensor::logical_to_tensor(&logical).map_err(|err| {
                            sample_error_with_detail(builtin, &SAMPLE_ERROR_INVALID_INPUT, err)
                        })?;
                        Self::from_tensor(tensor, builtin)
                    }
                    Value::ComplexTensor(tensor) => Ok(Self::Complex {
                        data: tensor.materialize_f64(),
                        shape: tensor.shape,
                    }),
                    other => Err(sample_error_with_detail(
                        builtin,
                        &SAMPLE_ERROR_INVALID_INPUT,
                        format!("gathered GPU value produced {other:?}"),
                    )),
                }
            }
            Value::Tensor(tensor) => Self::from_tensor(tensor, builtin),
            Value::LogicalArray(logical) => {
                let tensor = tensor::logical_to_tensor(&logical).map_err(|err| {
                    sample_error_with_detail(builtin, &SAMPLE_ERROR_INVALID_INPUT, err)
                })?;
                Self::from_tensor(tensor, builtin)
            }
            Value::ComplexTensor(tensor) => Ok(Self::Complex {
                data: tensor.materialize_f64(),
                shape: tensor.shape,
            }),
            Value::Num(value) => Ok(Self::Real {
                storage: NumericStorage::F64(vec![value]),
                shape: vec![1, 1],
            }),
            Value::Int(value) => Ok(Self::Real {
                storage: NumericStorage::from_integer_storage(IntegerStorage::from_scalar(value)),
                shape: vec![1, 1],
            }),
            Value::Bool(value) => Ok(Self::Real {
                storage: NumericStorage::F64(vec![if value { 1.0 } else { 0.0 }]),
                shape: vec![1, 1],
            }),
            Value::Complex(re, im) => Ok(Self::Complex {
                data: vec![(re, im)],
                shape: vec![1, 1],
            }),
            other => Err(sample_error_with_detail(
                builtin,
                &SAMPLE_ERROR_INVALID_INPUT,
                format!("received {other:?}"),
            )),
        }
    }

    fn from_tensor(tensor: Tensor, builtin: &'static str) -> BuiltinResult<Self> {
        let shape = tensor.shape.clone();
        let storage = tensor
            .into_numeric_storage()
            .map_err(|err| sample_error_with_detail(builtin, &SAMPLE_ERROR_INVALID_INPUT, err))?;
        Ok(Self::Real { storage, shape })
    }
}

fn sample_error(builtin: &'static str, error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    sample_error_with_message(builtin, error.message, error)
}

fn sample_error_with_detail(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    sample_error_with_message(
        builtin,
        format!("{}: {}", error.message, detail.as_ref()),
        error,
    )
}

fn sample_error_with_message(
    builtin: &'static str,
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sample_error_with_source(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(builtin)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "upsample",
    category = "math/signal",
    summary = "Increase sample rate by inserting zeros between samples.",
    keywords = "upsample,sample rate,zero insertion,signal processing",
    type_resolver(upsample_type),
    descriptor(crate::builtins::math::signal::sample_rate::UPSAMPLE_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::sample_rate"
)]
async fn upsample_builtin(x: Value, n: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    sample_rate_builtin(UPSAMPLE_NAME, SampleOp::Up, x, n, rest).await
}

#[runtime_builtin(
    name = "downsample",
    category = "math/signal",
    summary = "Decrease sample rate by keeping every Nth sample.",
    keywords = "downsample,sample rate,decimation,signal processing",
    type_resolver(downsample_type),
    extensions(DOWNSAMPLE_EXTENSIONS),
    integer_capabilities(DOWNSAMPLE_INTEGER_CAPABILITIES),
    descriptor(crate::builtins::math::signal::sample_rate::DOWNSAMPLE_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::sample_rate"
)]
async fn downsample_builtin(x: Value, n: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if is_typed_integer_value(&n) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DOWNSAMPLE_INTEGER_FACTOR_EXTENSION,
            DOWNSAMPLE_NAME,
        )?;
    }
    if rest.first().is_some_and(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DOWNSAMPLE_INTEGER_PHASE_EXTENSION,
            DOWNSAMPLE_NAME,
        )?;
    }
    if value_rank(&x).is_some_and(|rank| rank > 2) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DOWNSAMPLE_ND_INPUT_EXTENSION,
            DOWNSAMPLE_NAME,
        )?;
    }
    sample_rate_builtin(DOWNSAMPLE_NAME, SampleOp::Down, x, n, rest).await
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn value_rank(value: &Value) -> Option<usize> {
    match value {
        Value::Tensor(tensor) => Some(tensor.shape.len()),
        Value::LogicalArray(array) => Some(array.shape.len()),
        Value::ComplexTensor(tensor) => Some(tensor.shape.len()),
        Value::GpuTensor(handle) => Some(handle.shape.len()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Some(2),
        _ => None,
    }
}

#[runtime_builtin(
    name = "resample",
    category = "math/signal",
    summary = "Resample a signal by a rational factor with FIR antialiasing.",
    keywords = "resample,sample rate,polyphase,antialiasing,FIR,signal processing",
    type_resolver(resample_type),
    descriptor(crate::builtins::math::signal::sample_rate::RESAMPLE_DESCRIPTOR),
    extensions(RESAMPLE_EXTENSIONS),
    integer_capabilities(RESAMPLE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::signal::sample_rate"
)]
async fn resample_builtin(x: Value, p: Value, q: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count();
    match requested_outputs {
        Some(0) => return Ok(Value::OutputList(Vec::new())),
        Some(count) if count > 2 => {
            return Err(sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_OUTPUT_COUNT));
        }
        _ => {}
    }

    if is_typed_integer_value(&x) {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_INPUT,
            "resample input must be single or double",
        ));
    }
    let typed_options = is_typed_integer_value(&p)
        || is_typed_integer_value(&q)
        || rest.iter().any(is_typed_integer_value);
    if typed_options {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RESAMPLE_INTEGER_OPTION_EXTENSION,
            RESAMPLE_NAME,
        )?;
    }
    let mut p = parse_factor(&p, RESAMPLE_NAME).await?;
    let mut q = parse_factor(&q, RESAMPLE_NAME).await?;
    let divisor = gcd(p, q);
    p /= divisor;
    q /= divisor;
    let options = parse_resample_options(rest, p, q).await?;
    if let Value::GpuTensor(handle) = &x {
        if let Some(eval) = resample_gpu(handle, p, q, &options).await? {
            return match requested_outputs {
                Some(1) => Ok(Value::OutputList(vec![eval.y])),
                Some(2) => Ok(Value::OutputList(vec![
                    eval.y,
                    Value::Tensor(filter_tensor(eval.filter.host_values().await?)?),
                ])),
                None => Ok(eval.y),
                _ => unreachable!("output count was validated before evaluation"),
            };
        }
    }
    let input = SampleInput::from_value(x, RESAMPLE_NAME).await?;
    let host_options = HostResampleOptions {
        filter: options.filter.host_values().await?,
        dim: options.dim,
    };
    let eval = apply_resample(input, p, q, &host_options)?;
    match requested_outputs {
        Some(1) => Ok(Value::OutputList(vec![eval.y])),
        Some(2) => Ok(Value::OutputList(vec![
            eval.y,
            Value::Tensor(filter_tensor(eval.filter.host_values().await?)?),
        ])),
        None => Ok(eval.y),
        _ => unreachable!("output count was validated before evaluation"),
    }
}

async fn sample_rate_builtin(
    builtin: &'static str,
    op: SampleOp,
    x: Value,
    n: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(sample_error(builtin, &SAMPLE_ERROR_ARG_COUNT));
    }
    let factor = parse_factor(&n, builtin).await?;
    let phase = match rest.first() {
        Some(value) => parse_phase(value, factor, builtin).await?,
        None => 0,
    };
    if let Value::GpuTensor(handle) = &x {
        match op {
            SampleOp::Up => {
                if let Some(output) = upsample_gpu(handle, factor, phase, builtin)? {
                    return Ok(output);
                }
            }
            SampleOp::Down => {
                if let Some(output) = downsample_gpu(handle, factor, phase, builtin)? {
                    return Ok(output);
                }
            }
        }
    }
    let input = SampleInput::from_value(x, builtin).await?;
    apply_sample_rate(input, factor, phase, op, builtin)
}

#[derive(Clone, Debug)]
struct ResampleOptions {
    filter: ResampleFilter,
    dim: Option<usize>,
}

#[derive(Clone, Debug)]
enum ResampleFilter {
    Host(Vec<f64>),
    Gpu {
        handle: runmat_accelerate_api::GpuTensorHandle,
        len: usize,
    },
}

impl ResampleFilter {
    fn len(&self) -> usize {
        match self {
            Self::Host(filter) => filter.len(),
            Self::Gpu { len, .. } => *len,
        }
    }

    fn same_provider_handle(
        &self,
        source: &runmat_accelerate_api::GpuTensorHandle,
    ) -> Option<&runmat_accelerate_api::GpuTensorHandle> {
        match self {
            Self::Gpu { handle, .. } if handle.device_id == source.device_id => Some(handle),
            _ => None,
        }
    }

    async fn host_values(&self) -> BuiltinResult<Vec<f64>> {
        match self {
            Self::Host(filter) => Ok(filter.clone()),
            Self::Gpu { handle, .. } => {
                let tensor = gpu_helpers::gather_tensor_async(handle)
                    .await
                    .map_err(|err| {
                        sample_error_with_detail(
                            RESAMPLE_NAME,
                            &SAMPLE_ERROR_GATHER_FAILED,
                            err.message(),
                        )
                    })?;
                Ok(tensor::tensor_into_values_f64(tensor))
            }
        }
    }
}

#[derive(Clone, Debug)]
struct HostResampleOptions {
    filter: Vec<f64>,
    dim: Option<usize>,
}

struct ResampleEval {
    y: Value,
    filter: ResampleFilter,
}

async fn resample_gpu(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    p: usize,
    q: usize,
    options: &ResampleOptions,
) -> BuiltinResult<Option<ResampleEval>> {
    let Some(provider) = runmat_accelerate_api::provider_for_handle(handle) else {
        return Ok(None);
    };
    let storage = runmat_accelerate_api::handle_storage(handle);
    let handle_len = checked_product(&handle.shape)
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut shape = canonical_shape(handle.shape.clone(), handle_len);
    let dim = options
        .dim
        .unwrap_or_else(|| first_non_singleton_dim(&shape));
    if dim >= shape.len() {
        shape.resize(dim + 1, 1);
    }

    let input_len = shape[dim];
    let output_len = resample_output_len(input_len, p, q)
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output_shape = shape.clone();
    output_shape[dim] = output_len;

    if input_len == 0 || output_len == 0 {
        return match provider.zeros_with_storage(&output_shape, storage) {
            Ok(output) => Ok(Some(ResampleEval {
                y: wrap_resampled_gpu(handle, output),
                filter: options.filter.clone(),
            })),
            Err(_) => Ok(None),
        };
    }

    let filter_len = options.filter.len();
    let delay = filter_len / 2;
    let gather_span_len = output_len
        .checked_sub(1)
        .and_then(|last| last.checked_mul(q))
        .and_then(|last| last.checked_add(delay))
        .and_then(|last| last.checked_add(1))
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let scatter_span_len = input_len
        .checked_sub(1)
        .and_then(|last| last.checked_mul(p))
        .and_then(|last| last.checked_add(1))
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let padded_len = gather_span_len.max(scatter_span_len);
    let mut padded_shape = shape.clone();
    padded_shape[dim] = padded_len;

    let Some(scatter_indices) =
        upsample_linear_indices(&shape, dim, input_len, padded_len, p, 0, RESAMPLE_NAME)?
    else {
        return Ok(None);
    };
    let Some(gather_indices) = downsample_linear_indices(
        &padded_shape,
        dim,
        padded_len,
        output_len,
        q,
        delay,
        RESAMPLE_NAME,
    )?
    else {
        return Ok(None);
    };

    let padded = match provider.zeros_with_storage(&padded_shape, storage) {
        Ok(handle) => handle,
        Err(_) => return Ok(None),
    };
    if provider
        .scatter_linear(&padded, &scatter_indices, handle)
        .is_err()
    {
        let _ = provider.free(&padded);
        return Ok(None);
    }

    let filter_shape = [filter_len, 1usize];
    let (b_handle, free_b_handle) =
        if let Some(handle) = options.filter.same_provider_handle(handle) {
            (handle.clone(), false)
        } else {
            let filter = options.filter.host_values().await?;
            match provider.upload(&runmat_accelerate_api::HostTensorView {
                data: &filter,
                shape: &filter_shape,
            }) {
                Ok(handle) => (handle, true),
                Err(_) => {
                    let _ = provider.free(&padded);
                    return Ok(None);
                }
            }
        };
    let a_coeff = [1.0f64];
    let a_shape = [1usize, 1usize];
    let a_handle = match provider.upload(&runmat_accelerate_api::HostTensorView {
        data: &a_coeff,
        shape: &a_shape,
    }) {
        Ok(handle) => handle,
        Err(_) => {
            if free_b_handle {
                let _ = provider.free(&b_handle);
            }
            let _ = provider.free(&padded);
            return Ok(None);
        }
    };

    let filter_result = match provider
        .iir_filter(
            &b_handle,
            &a_handle,
            &padded,
            runmat_accelerate_api::ProviderIirFilterOptions {
                dim,
                zi: None,
                unit_denominator: true,
            },
        )
        .await
    {
        Ok(result) => result,
        Err(_) => {
            let _ = provider.free(&a_handle);
            if free_b_handle {
                let _ = provider.free(&b_handle);
            }
            let _ = provider.free(&padded);
            return Ok(None);
        }
    };

    let output = provider.gather_linear(&filter_result.output, &gather_indices, &output_shape);
    if let Some(final_state) = &filter_result.final_state {
        let _ = provider.free(final_state);
    }
    let _ = provider.free(&filter_result.output);
    let _ = provider.free(&a_handle);
    if free_b_handle {
        let _ = provider.free(&b_handle);
    }
    let _ = provider.free(&padded);

    match output {
        Ok(output) => Ok(Some(ResampleEval {
            y: wrap_resampled_gpu(handle, output),
            filter: options.filter.clone(),
        })),
        Err(_) => Ok(None),
    }
}

async fn parse_resample_options(
    rest: Vec<Value>,
    p: usize,
    q: usize,
) -> BuiltinResult<ResampleOptions> {
    let mut idx = 0usize;
    let mut positional = Vec::new();
    let mut dim = None;
    while idx < rest.len() {
        if let Some(key) = value_as_text(&rest[idx]) {
            if key.eq_ignore_ascii_case("dimension") {
                let Some(value) = rest.get(idx + 1) else {
                    return Err(sample_error_with_detail(
                        RESAMPLE_NAME,
                        &SAMPLE_ERROR_INVALID_OPTION,
                        "Dimension name-value option requires a value",
                    ));
                };
                dim = Some(
                    tensor::parse_dimension(value, RESAMPLE_NAME)
                        .map_err(|err| {
                            sample_error_with_detail(
                                RESAMPLE_NAME,
                                &SAMPLE_ERROR_INVALID_OPTION,
                                err,
                            )
                        })?
                        .saturating_sub(1),
                );
                idx += 2;
                continue;
            }
        }
        positional.push(rest[idx].clone());
        idx += 1;
    }

    let filter = if positional.is_empty() {
        ResampleFilter::Host(design_resample_filter(
            p,
            q,
            DEFAULT_RESAMPLE_N,
            DEFAULT_RESAMPLE_BETA,
        )?)
    } else {
        parse_resample_filter_spec(&positional, p, q).await?
    };
    Ok(ResampleOptions { filter, dim })
}

async fn parse_resample_filter_spec(
    args: &[Value],
    p: usize,
    q: usize,
) -> BuiltinResult<ResampleFilter> {
    match args.len() {
        0 => Ok(ResampleFilter::Host(design_resample_filter(
            p,
            q,
            DEFAULT_RESAMPLE_N,
            DEFAULT_RESAMPLE_BETA,
        )?)),
        1 => {
            if let Some(n) = scalar_integer_option(&args[0]).await? {
                Ok(ResampleFilter::Host(design_resample_filter(
                    p,
                    q,
                    n,
                    DEFAULT_RESAMPLE_BETA,
                )?))
            } else {
                parse_filter_vector(args[0].clone()).await
            }
        }
        2 => {
            let Some(n) = scalar_integer_option(&args[0]).await? else {
                return Err(sample_error_with_detail(
                    RESAMPLE_NAME,
                    &SAMPLE_ERROR_INVALID_OPTION,
                    "N must be a nonnegative integer scalar when BETA is provided",
                ));
            };
            let beta = scalar_f64_option(&args[1]).await?;
            Ok(ResampleFilter::Host(design_resample_filter(p, q, n, beta)?))
        }
        _ => Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_ARG_COUNT,
            "resample accepts X, P, Q, optional N/BETA or B, and optional Dimension",
        )),
    }
}

async fn scalar_integer_option(value: &Value) -> BuiltinResult<Option<usize>> {
    match value {
        Value::Tensor(tensor) if !tensor::is_scalar_tensor(tensor) => return Ok(None),
        Value::GpuTensor(handle) if checked_product(&handle.shape) != Some(1) => return Ok(None),
        Value::ComplexTensor(_) => return Ok(None),
        _ => {}
    }
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return parse_nonnegative_integer_value(
            integer,
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
        )
        .map(Some);
    }
    let Some(raw) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|err| {
            sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_INVALID_OPTION, err)
        })?
    else {
        return Ok(None);
    };
    parse_nonnegative_integer(raw, RESAMPLE_NAME, &SAMPLE_ERROR_INVALID_OPTION).map(Some)
}

async fn scalar_f64_option(value: &Value) -> BuiltinResult<f64> {
    if is_typed_integer_value(value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
            .await?
    {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "typed integer BETA must be exactly representable as double",
        ));
    }
    let Some(raw) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|err| {
            sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_INVALID_OPTION, err)
        })?
    else {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            format!("expected scalar option, received {value:?}"),
        ));
    };
    if !raw.is_finite() || raw < 0.0 {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "BETA must be a finite nonnegative scalar",
        ));
    }
    Ok(raw)
}

async fn parse_filter_vector(value: Value) -> BuiltinResult<ResampleFilter> {
    if is_typed_integer_value(&value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(&value)
            .await?
    {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "typed integer filter coefficients must be exactly representable as double",
        ));
    }
    if let Value::GpuTensor(handle) = value {
        let len = checked_product(&handle.shape)
            .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
        validate_filter_shape(len, &handle.shape)?;
        if runmat_accelerate_api::handle_storage(&handle)
            != runmat_accelerate_api::GpuTensorStorage::Real
        {
            return Err(sample_error_with_detail(
                RESAMPLE_NAME,
                &SAMPLE_ERROR_INVALID_OPTION,
                "filter coefficients must be real-valued",
            ));
        }
        return Ok(ResampleFilter::Gpu { handle, len });
    }

    let tensor = tensor::value_into_tensor_for(RESAMPLE_NAME, value).map_err(|err| {
        sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_INVALID_OPTION, err)
    })?;
    let shape = tensor.shape.clone();
    let values = tensor::tensor_into_values_f64(tensor);
    validate_filter_shape(values.len(), &shape)?;
    Ok(ResampleFilter::Host(values))
}

fn validate_filter_shape(len: usize, shape: &[usize]) -> BuiltinResult<()> {
    if len == 0 {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "filter coefficient vector cannot be empty",
        ));
    }
    if shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "filter coefficients must be a vector",
        ));
    }
    Ok(())
}

fn apply_resample(
    input: SampleInput,
    p: usize,
    q: usize,
    options: &HostResampleOptions,
) -> BuiltinResult<ResampleEval> {
    match input {
        SampleInput::Real { storage, shape } => {
            let dtype = storage.numeric_dtype();
            let data = match storage {
                NumericStorage::F64(values) => values,
                NumericStorage::F32(values) => {
                    values.into_iter().map(f64::from).collect::<Vec<_>>()
                }
                NumericStorage::I8(_)
                | NumericStorage::I16(_)
                | NumericStorage::I32(_)
                | NumericStorage::I64(_)
                | NumericStorage::U8(_)
                | NumericStorage::U16(_)
                | NumericStorage::U32(_)
                | NumericStorage::U64(_) => {
                    return Err(sample_error_with_detail(
                        RESAMPLE_NAME,
                        &SAMPLE_ERROR_INVALID_INPUT,
                        "resample input must be single or double",
                    ))
                }
            };
            let (output, shape) = resample_rational_column_major(data, shape, p, q, options)?;
            let storage = match dtype {
                runmat_builtins::NumericDType::F64 => NumericStorage::F64(output),
                runmat_builtins::NumericDType::F32 => {
                    NumericStorage::F32(output.into_iter().map(|value| value as f32).collect())
                }
                _ => unreachable!("resample input dtype was validated"),
            };
            let tensor = Tensor::from_numeric_storage(storage, shape).map_err(|err| {
                sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_BUILD_OUTPUT, err)
            })?;
            Ok(ResampleEval {
                y: tensor::tensor_into_value(tensor),
                filter: ResampleFilter::Host(options.filter.clone()),
            })
        }
        SampleInput::Complex { data, shape } => {
            let (output, shape) = resample_rational_column_major(data, shape, p, q, options)?;
            let tensor = ComplexTensor::new(output, shape).map_err(|err| {
                sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_BUILD_OUTPUT, err)
            })?;
            let y = if tensor.materialize_f64().len() == 1 {
                let (re, im) = tensor.materialize_f64()[0];
                Value::Complex(re, im)
            } else {
                Value::ComplexTensor(tensor)
            };
            Ok(ResampleEval {
                y,
                filter: ResampleFilter::Host(options.filter.clone()),
            })
        }
    }
}

async fn parse_factor(value: &Value, builtin: &'static str) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return parse_positive_integer_value(integer, builtin, &SAMPLE_ERROR_INVALID_FACTOR);
    }
    let Some(raw) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|detail| {
            sample_error_with_detail(builtin, &SAMPLE_ERROR_INVALID_FACTOR, detail)
        })?
    else {
        return Err(sample_error_with_detail(
            builtin,
            &SAMPLE_ERROR_INVALID_FACTOR,
            format!("received {value:?}"),
        ));
    };
    parse_positive_integer(raw, builtin, &SAMPLE_ERROR_INVALID_FACTOR)
}

async fn parse_phase(value: &Value, factor: usize, builtin: &'static str) -> BuiltinResult<usize> {
    let phase = if let Some(integer) = tensor::scalar_integer_value(value) {
        parse_nonnegative_integer_value(integer, builtin, &SAMPLE_ERROR_INVALID_PHASE)?
    } else {
        let Some(raw) = tensor::scalar_f64_from_value_async(value)
            .await
            .map_err(|detail| {
                sample_error_with_detail(builtin, &SAMPLE_ERROR_INVALID_PHASE, detail)
            })?
        else {
            return Err(sample_error_with_detail(
                builtin,
                &SAMPLE_ERROR_INVALID_PHASE,
                format!("received {value:?}"),
            ));
        };
        parse_nonnegative_integer(raw, builtin, &SAMPLE_ERROR_INVALID_PHASE)?
    };
    if phase >= factor {
        return Err(sample_error_with_detail(
            builtin,
            &SAMPLE_ERROR_INVALID_PHASE,
            format!("phase {phase} is not less than N ({factor})"),
        ));
    }
    Ok(phase)
}

fn parse_positive_integer(
    raw: f64,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<usize> {
    let value = parse_nonnegative_integer(raw, builtin, error)?;
    if value == 0 {
        return Err(sample_error(builtin, error));
    }
    Ok(value)
}

fn parse_positive_integer_value(
    value: runmat_builtins::IntValue,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<usize> {
    let value = parse_nonnegative_integer_value(value, builtin, error)?;
    if value == 0 {
        return Err(sample_error(builtin, error));
    }
    Ok(value)
}

fn parse_nonnegative_integer_value(
    value: runmat_builtins::IntValue,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .ok_or_else(|| sample_error(builtin, error))
}

fn parse_nonnegative_integer(
    raw: f64,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<usize> {
    if !raw.is_finite() || raw < 0.0 {
        return Err(sample_error(builtin, error));
    }
    if raw.trunc() != raw
        || raw > usize::MAX as f64
        || (usize::BITS == 64 && raw == usize::MAX as f64)
    {
        return Err(sample_error(builtin, error));
    }
    Ok(raw as usize)
}

fn apply_sample_rate(
    input: SampleInput,
    factor: usize,
    phase: usize,
    op: SampleOp,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match input {
        SampleInput::Real { storage, shape } => {
            let (storage, shape) =
                resample_numeric_storage(storage, shape, factor, phase, op, builtin)?;
            let tensor = Tensor::from_numeric_storage(storage, shape).map_err(|err| {
                sample_error_with_detail(builtin, &SAMPLE_ERROR_BUILD_OUTPUT, err)
            })?;
            Ok(tensor::tensor_into_value(tensor))
        }
        SampleInput::Complex { data, shape } => {
            let (output, shape) =
                resample_column_major(data, shape, factor, phase, op, (0.0, 0.0), builtin)?;
            let tensor = ComplexTensor::new(output, shape).map_err(|err| {
                sample_error_with_detail(builtin, &SAMPLE_ERROR_BUILD_OUTPUT, err)
            })?;
            if tensor.materialize_f64().len() == 1 {
                let (re, im) = tensor.materialize_f64()[0];
                Ok(Value::Complex(re, im))
            } else {
                Ok(Value::ComplexTensor(tensor))
            }
        }
    }
}

fn resample_numeric_storage(
    storage: NumericStorage,
    shape: Vec<usize>,
    factor: usize,
    phase: usize,
    op: SampleOp,
    builtin: &'static str,
) -> BuiltinResult<(NumericStorage, Vec<usize>)> {
    macro_rules! resample {
        ($values:expr, $zero:expr, $variant:ident) => {{
            let (values, shape) =
                resample_column_major($values, shape, factor, phase, op, $zero, builtin)?;
            (NumericStorage::$variant(values), shape)
        }};
    }
    Ok(match storage {
        NumericStorage::F64(values) => resample!(values, 0.0, F64),
        NumericStorage::F32(values) => resample!(values, 0.0_f32, F32),
        NumericStorage::I8(values) => resample!(values, 0_i8, I8),
        NumericStorage::I16(values) => resample!(values, 0_i16, I16),
        NumericStorage::I32(values) => resample!(values, 0_i32, I32),
        NumericStorage::I64(values) => resample!(values, 0_i64, I64),
        NumericStorage::U8(values) => resample!(values, 0_u8, U8),
        NumericStorage::U16(values) => resample!(values, 0_u16, U16),
        NumericStorage::U32(values) => resample!(values, 0_u32, U32),
        NumericStorage::U64(values) => resample!(values, 0_u64, U64),
    })
}

fn upsample_gpu(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    factor: usize,
    phase: usize,
    builtin: &'static str,
) -> BuiltinResult<Option<Value>> {
    let handle_len = checked_product(&handle.shape)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let shape = canonical_shape(handle.shape.clone(), handle_len);
    let dim = first_non_singleton_dim(&shape);
    let input_len = shape[dim];
    let output_len = output_len(input_len, factor, phase, SampleOp::Up)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output_shape = shape.clone();
    output_shape[dim] = output_len;

    if factor == 1 && phase == 0 {
        return Ok(Some(wrap_sample_rate_gpu(handle, handle.clone())));
    }

    let Some(provider) = runmat_accelerate_api::provider_for_handle(handle) else {
        return Ok(None);
    };
    let Some(indices) =
        upsample_linear_indices(&shape, dim, input_len, output_len, factor, phase, builtin)?
    else {
        return Ok(None);
    };
    let allocation = match runmat_accelerate_api::handle_integer_type(handle) {
        Some(_) => provider.zeros_integer_like(handle, &output_shape),
        None => provider
            .zeros_with_storage(&output_shape, runmat_accelerate_api::handle_storage(handle)),
    };
    let output = match allocation {
        Ok(output) => output,
        Err(_) => return Ok(None),
    };
    match provider.scatter_linear(&output, &indices, handle) {
        Ok(()) => Ok(Some(wrap_sample_rate_gpu(handle, output))),
        Err(_) => {
            let _ = provider.free(&output);
            Ok(None)
        }
    }
}

fn downsample_gpu(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    factor: usize,
    phase: usize,
    builtin: &'static str,
) -> BuiltinResult<Option<Value>> {
    let handle_len = checked_product(&handle.shape)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let shape = canonical_shape(handle.shape.clone(), handle_len);
    let dim = first_non_singleton_dim(&shape);
    let input_len = shape[dim];
    let output_len = output_len(input_len, factor, phase, SampleOp::Down)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output_shape = shape.clone();
    output_shape[dim] = output_len;

    let provider = resolved_actual_downsample_owner(handle).ok_or_else(|| {
        sample_error_with_detail(
            builtin,
            &SAMPLE_ERROR_GATHER_FAILED,
            "GPU input has no proven owner for its device",
        )
    })?;

    if factor == 1 && phase == 0 {
        return Ok(Some(wrap_sample_rate_gpu(handle, handle.clone())));
    }

    let Some(indices) =
        downsample_linear_indices(&shape, dim, input_len, output_len, factor, phase, builtin)?
    else {
        return Ok(None);
    };

    match provider.gather_linear(handle, &indices, &output_shape) {
        Ok(output) => {
            let output_identity = (output.device_id, output.buffer_id);
            let input_identity = (handle.device_id, handle.buffer_id);
            let valid = output_identity != input_identity
                && output.shape == output_shape
                && output.device_id == handle.device_id
                && resolved_actual_downsample_owner(&output)
                    .is_some_and(|owner| std::ptr::eq(owner, provider))
                && runmat_accelerate_api::handle_storage(&output)
                    == runmat_accelerate_api::handle_storage(handle)
                && runmat_accelerate_api::handle_integer_type(&output)
                    == runmat_accelerate_api::handle_integer_type(handle)
                && runmat_accelerate_api::handle_is_logical(&output)
                    == runmat_accelerate_api::handle_is_logical(handle)
                && (runmat_accelerate_api::handle_integer_type(handle).is_some()
                    || runmat_accelerate_api::handle_precision(&output)
                        == runmat_accelerate_api::handle_precision(handle));
            if valid {
                Ok(Some(wrap_sample_rate_gpu(handle, output)))
            } else {
                if output_identity != input_identity {
                    if let Some(owner) = resolved_actual_downsample_owner(&output) {
                        let _ = owner.free(&output);
                    }
                }
                Err(sample_error_with_detail(
                    builtin,
                    &SAMPLE_ERROR_GATHER_FAILED,
                    "provider returned an incompatible linear-gather result",
                ))
            }
        }
        Err(_) => Ok(None),
    }
}

fn resolved_actual_downsample_owner(
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> Option<&'static dyn runmat_accelerate_api::AccelProvider> {
    runmat_accelerate_api::provider_for_handle(handle)
        .filter(|owner| owner.device_id() == handle.device_id)
}

fn wrap_sample_rate_gpu(
    source: &runmat_accelerate_api::GpuTensorHandle,
    output: runmat_accelerate_api::GpuTensorHandle,
) -> Value {
    if let Some(precision) = runmat_accelerate_api::handle_precision(source) {
        runmat_accelerate_api::set_handle_precision(&output, precision);
    }
    if runmat_accelerate_api::handle_is_logical(source) {
        gpu_helpers::logical_gpu_value(output)
    } else {
        runmat_accelerate_api::set_handle_storage(
            &output,
            runmat_accelerate_api::handle_storage(source),
        );
        gpu_helpers::resident_gpu_value(output)
    }
}

fn wrap_resampled_gpu(
    source: &runmat_accelerate_api::GpuTensorHandle,
    output: runmat_accelerate_api::GpuTensorHandle,
) -> Value {
    if let Some(precision) = runmat_accelerate_api::handle_precision(source) {
        runmat_accelerate_api::set_handle_precision(&output, precision);
    }
    runmat_accelerate_api::set_handle_storage(
        &output,
        runmat_accelerate_api::handle_storage(source),
    );
    runmat_accelerate_api::clear_handle_logical(&output);
    gpu_helpers::resident_gpu_value(output)
}

fn upsample_linear_indices(
    shape: &[usize],
    dim: usize,
    input_len: usize,
    output_len: usize,
    factor: usize,
    phase: usize,
    builtin: &'static str,
) -> BuiltinResult<Option<Vec<u32>>> {
    let input_count = checked_element_count(shape)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    if input_count > u32::MAX as usize {
        return Ok(None);
    }

    let leading = checked_product(&shape[..dim])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let trailing = checked_product(&shape[dim + 1..])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(input_count)
        .map_err(|_| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;

    for trail in 0..trailing {
        for input_pos in 0..input_len {
            let output_pos = input_pos
                .checked_mul(factor)
                .and_then(|offset| offset.checked_add(phase))
                .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
            let trail_offset = output_len
                .checked_mul(trail)
                .and_then(|offset| offset.checked_add(output_pos))
                .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
            for before in 0..leading {
                let Some(dst) = before.checked_add(
                    leading
                        .checked_mul(trail_offset)
                        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?,
                ) else {
                    return Err(sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW));
                };
                let Ok(dst) = u32::try_from(dst) else {
                    return Ok(None);
                };
                indices.push(dst);
            }
        }
    }

    Ok(Some(indices))
}

fn downsample_linear_indices(
    shape: &[usize],
    dim: usize,
    input_len: usize,
    output_len: usize,
    factor: usize,
    phase: usize,
    builtin: &'static str,
) -> BuiltinResult<Option<Vec<u32>>> {
    let output_shape_count = {
        let mut output_shape = shape.to_vec();
        output_shape[dim] = output_len;
        checked_element_count(&output_shape)
            .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?
    };
    if output_shape_count > u32::MAX as usize {
        return Ok(None);
    }

    let leading = checked_product(&shape[..dim])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let trailing = checked_product(&shape[dim + 1..])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(output_shape_count)
        .map_err(|_| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;

    for trail in 0..trailing {
        for output_pos in 0..output_len {
            let input_pos = output_pos
                .checked_mul(factor)
                .and_then(|offset| offset.checked_add(phase))
                .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
            let trail_offset = input_len
                .checked_mul(trail)
                .and_then(|offset| offset.checked_add(input_pos))
                .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
            for before in 0..leading {
                let Some(src) = before.checked_add(
                    leading
                        .checked_mul(trail_offset)
                        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?,
                ) else {
                    return Err(sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW));
                };
                let Ok(src) = u32::try_from(src) else {
                    return Ok(None);
                };
                indices.push(src);
            }
        }
    }

    Ok(Some(indices))
}

fn resample_column_major<T: Copy>(
    data: Vec<T>,
    shape: Vec<usize>,
    factor: usize,
    phase: usize,
    op: SampleOp,
    zero: T,
    builtin: &'static str,
) -> BuiltinResult<(Vec<T>, Vec<usize>)> {
    let shape = canonical_shape(shape, data.len());
    let dim = first_non_singleton_dim(&shape);
    let input_len = shape[dim];
    let output_len = output_len(input_len, factor, phase, op)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output_shape = shape.clone();
    output_shape[dim] = output_len;
    let output_count = checked_element_count(&output_shape)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_count)
        .map_err(|_| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    output.resize(output_count, zero);
    if input_len == 0 || output_len == 0 {
        return Ok((output, output_shape));
    }

    let leading = checked_product(&shape[..dim])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let trailing = checked_product(&shape[dim + 1..])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;

    for trail in 0..trailing {
        for before in 0..leading {
            match op {
                SampleOp::Up => {
                    for input_pos in 0..input_len {
                        let output_pos = phase + input_pos * factor;
                        let src = before + leading * (input_pos + input_len * trail);
                        let dst = before + leading * (output_pos + output_len * trail);
                        output[dst] = data[src];
                    }
                }
                SampleOp::Down => {
                    for output_pos in 0..output_len {
                        let input_pos = phase + output_pos * factor;
                        let src = before + leading * (input_pos + input_len * trail);
                        let dst = before + leading * (output_pos + output_len * trail);
                        output[dst] = data[src];
                    }
                }
            }
        }
    }

    Ok((output, output_shape))
}

trait ResampleSample: Copy {
    fn zero() -> Self;
    fn add_scaled(self, sample: Self, coeff: f64) -> Self;
}

impl ResampleSample for f64 {
    fn zero() -> Self {
        0.0
    }

    fn add_scaled(self, sample: Self, coeff: f64) -> Self {
        self + sample * coeff
    }
}

impl ResampleSample for (f64, f64) {
    fn zero() -> Self {
        (0.0, 0.0)
    }

    fn add_scaled(self, sample: Self, coeff: f64) -> Self {
        (self.0 + sample.0 * coeff, self.1 + sample.1 * coeff)
    }
}

fn resample_rational_column_major<T: ResampleSample>(
    data: Vec<T>,
    shape: Vec<usize>,
    p: usize,
    q: usize,
    options: &HostResampleOptions,
) -> BuiltinResult<(Vec<T>, Vec<usize>)> {
    let mut shape = canonical_shape(shape, data.len());
    let dim = options
        .dim
        .unwrap_or_else(|| first_non_singleton_dim(&shape));
    if dim >= shape.len() {
        shape.resize(dim + 1, 1);
    }
    let input_len = shape[dim];
    let output_len = resample_output_len(input_len, p, q)
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output_shape = shape.clone();
    output_shape[dim] = output_len;
    let output_count = checked_element_count(&output_shape)
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_count)
        .map_err(|_| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    output.resize(output_count, T::zero());
    if input_len == 0 || output_len == 0 {
        return Ok((output, output_shape));
    }

    let leading = checked_product(&shape[..dim])
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let trailing = checked_product(&shape[dim + 1..])
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let delay = options.filter.len() / 2;

    for trail in 0..trailing {
        for before in 0..leading {
            for output_pos in 0..output_len {
                let Some(center) = output_pos
                    .checked_mul(q)
                    .and_then(|value| value.checked_add(delay))
                else {
                    return Err(sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW));
                };
                let mut acc = T::zero();
                for (tap, coeff) in options.filter.iter().enumerate() {
                    if tap > center {
                        break;
                    }
                    let upsampled_pos = center - tap;
                    if upsampled_pos % p != 0 {
                        continue;
                    }
                    let input_pos = upsampled_pos / p;
                    if input_pos >= input_len {
                        continue;
                    }
                    let src = before + leading * (input_pos + input_len * trail);
                    acc = acc.add_scaled(data[src], *coeff);
                }
                let dst = before + leading * (output_pos + output_len * trail);
                output[dst] = acc;
            }
        }
    }

    Ok((output, output_shape))
}

fn resample_output_len(input_len: usize, p: usize, q: usize) -> Option<usize> {
    input_len
        .checked_mul(p)?
        .checked_add(q.checked_sub(1)?)?
        .checked_div(q)
}

fn design_resample_filter(p: usize, q: usize, n: usize, beta: f64) -> BuiltinResult<Vec<f64>> {
    let max_factor = p.max(q);
    let half_len = n
        .checked_mul(max_factor)
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let len = half_len
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let cutoff = 1.0 / max_factor as f64;
    let mut filter = Vec::new();
    filter
        .try_reserve_exact(len)
        .map_err(|_| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let denom = modified_bessel_i0(beta);
    for idx in 0..len {
        let centered = idx as isize - half_len as isize;
        let sinc = if centered == 0 {
            cutoff
        } else {
            let x = std::f64::consts::PI * cutoff * centered as f64;
            x.sin() / (std::f64::consts::PI * centered as f64)
        };
        let window = if half_len == 0 {
            1.0
        } else {
            let ratio = centered as f64 / half_len as f64;
            modified_bessel_i0(beta * (1.0 - ratio * ratio).max(0.0).sqrt()) / denom
        };
        filter.push(sinc * window);
    }
    let sum = filter.iter().sum::<f64>();
    if !sum.is_finite() || sum == 0.0 {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "designed filter has zero or nonfinite gain",
        ));
    }
    let gain = p as f64 / sum;
    for coeff in &mut filter {
        *coeff *= gain;
    }
    Ok(filter)
}

fn modified_bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.515_622_9
                + y * (3.089_942_4
                    + y * (1.206_749_2 + y * (0.265_973_2 + y * (0.036_076_8 + y * 0.004_581_3)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.398_942_28
                + y * (0.013_285_92
                    + y * (0.002_253_19
                        + y * (-0.001_575_65
                            + y * (0.009_162_81
                                + y * (-0.020_577_06
                                    + y * (0.026_355_37
                                        + y * (-0.016_476_33 + y * 0.003_923_77))))))))
    }
}

fn filter_tensor(filter: Vec<f64>) -> BuiltinResult<Tensor> {
    let len = filter.len();
    Tensor::new(filter, vec![1, len])
        .map_err(|err| sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_BUILD_OUTPUT, err))
}

fn value_as_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        _ => None,
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a.max(1)
}

fn canonical_shape(shape: Vec<usize>, len: usize) -> Vec<usize> {
    if shape.is_empty() {
        if len == 0 {
            vec![0, 1]
        } else {
            vec![1, 1]
        }
    } else {
        shape
    }
}

fn first_non_singleton_dim(shape: &[usize]) -> usize {
    shape.iter().position(|dim| *dim != 1).unwrap_or(0)
}

fn output_len(input_len: usize, factor: usize, phase: usize, op: SampleOp) -> Option<usize> {
    match op {
        SampleOp::Up => input_len.checked_mul(factor),
        SampleOp::Down => {
            if input_len <= phase {
                Some(0)
            } else {
                Some(((input_len - 1 - phase) / factor) + 1)
            }
        }
    }
}

fn checked_element_count(shape: &[usize]) -> Option<usize> {
    checked_product(shape)
}

fn checked_product(values: &[usize]) -> Option<usize> {
    values
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorOwned,
        HostTensorView, IntegerElementType, ProviderPrecision,
    };
    use runmat_builtins::IntegerStorage;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
    use std::sync::Mutex;

    struct HostileDownsampleProvider {
        device_id: u32,
        next_buffer: AtomicU64,
        mode: AtomicU8,
        frees: AtomicUsize,
        downloads: AtomicUsize,
        gathers: AtomicUsize,
        last_output: AtomicU64,
        buffers: Mutex<HashMap<u64, HostTensorOwned>>,
    }

    impl HostileDownsampleProvider {
        fn new() -> Self {
            Self {
                device_id: runmat_accelerate_api::next_device_id(),
                next_buffer: AtomicU64::new(9_399_100_000),
                mode: AtomicU8::new(0),
                frees: AtomicUsize::new(0),
                downloads: AtomicUsize::new(0),
                gathers: AtomicUsize::new(0),
                last_output: AtomicU64::new(0),
                buffers: Mutex::new(HashMap::new()),
            }
        }

        fn allocate(
            &self,
            data: Vec<f64>,
            shape: Vec<usize>,
            device_id: u32,
            precision: ProviderPrecision,
            storage: GpuTensorStorage,
        ) -> GpuTensorHandle {
            let buffer_id = self.next_buffer.fetch_add(1, Ordering::Relaxed);
            self.buffers.lock().unwrap().insert(
                buffer_id,
                HostTensorOwned {
                    data,
                    shape: shape.clone(),
                    storage,
                },
            );
            let handle = GpuTensorHandle {
                shape,
                device_id,
                buffer_id,
            };
            runmat_accelerate_api::set_handle_precision(&handle, precision);
            runmat_accelerate_api::set_handle_storage(&handle, storage);
            handle
        }
    }

    impl AccelProvider for HostileDownsampleProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Ok(self.allocate(
                host.data.to_vec(),
                host.shape.to_vec(),
                self.device_id,
                ProviderPrecision::F64,
                GpuTensorStorage::Real,
            ))
        }

        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            self.downloads.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move {
                self.buffers
                    .lock()
                    .unwrap()
                    .get(&handle.buffer_id)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("unknown hostile test buffer"))
            })
        }

        fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
            if self
                .buffers
                .lock()
                .unwrap()
                .remove(&handle.buffer_id)
                .is_some()
            {
                self.frees.fetch_add(1, Ordering::Relaxed);
            }
            runmat_accelerate_api::clear_handle_precision(handle);
            runmat_accelerate_api::clear_handle_storage(handle);
            runmat_accelerate_api::clear_handle_integer_type(handle);
            runmat_accelerate_api::clear_handle_logical(handle);
            Ok(())
        }

        fn device_info(&self) -> String {
            "hostile-downsample-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            self.device_id
        }

        fn gather_linear(
            &self,
            source: &GpuTensorHandle,
            _indices: &[u32],
            output_shape: &[usize],
        ) -> anyhow::Result<GpuTensorHandle> {
            self.gathers.fetch_add(1, Ordering::Relaxed);
            let mode = self.mode.load(Ordering::Relaxed);
            if mode == 6 {
                return Err(anyhow::anyhow!("forced gather fallback"));
            }
            if mode == 7 {
                return Ok(source.clone());
            }
            let shape = if mode == 0 {
                vec![1, output_shape.iter().product::<usize>() + 1]
            } else {
                output_shape.to_vec()
            };
            let device_id = if mode == 1 {
                self.device_id.wrapping_add(10_000)
            } else {
                self.device_id
            };
            let precision = if mode == 3 {
                ProviderPrecision::F32
            } else {
                ProviderPrecision::F64
            };
            let storage = if mode == 2 {
                GpuTensorStorage::ComplexInterleaved
            } else {
                GpuTensorStorage::Real
            };
            let output = self.allocate(
                vec![99.0; shape.iter().product()],
                shape,
                device_id,
                precision,
                storage,
            );
            if mode == 4 {
                runmat_accelerate_api::set_handle_integer_type(&output, IntegerElementType::U8);
            }
            if mode == 5 {
                runmat_accelerate_api::set_handle_logical(&output, true);
            }
            self.last_output.store(output.buffer_id, Ordering::Relaxed);
            Ok(output)
        }
    }

    fn first_unrepresentable_usize_double() -> f64 {
        if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        }
    }

    fn exact_platform_wide_integer() -> u64 {
        if usize::BITS == 64 {
            9_007_199_254_740_993
        } else {
            u32::MAX as u64
        }
    }

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn call_upsample(args: Vec<Value>) -> Value {
        block_on(upsample_builtin(
            args[0].clone(),
            args[1].clone(),
            args[2..].to_vec(),
        ))
        .unwrap()
    }

    fn call_downsample(args: Vec<Value>) -> Value {
        block_on(downsample_builtin(
            args[0].clone(),
            args[1].clone(),
            args[2..].to_vec(),
        ))
        .unwrap()
    }

    fn call_resample(args: Vec<Value>) -> Value {
        block_on(resample_builtin(
            args[0].clone(),
            args[1].clone(),
            args[2].clone(),
            args[3..].to_vec(),
        ))
        .unwrap()
    }

    #[test]
    fn upsample_row_vector_inserts_zeros_after_each_sample() {
        let out = call_upsample(vec![
            tensor(vec![1.0, 2.0, 3.0], vec![1, 3]),
            Value::Num(2.0),
        ]);
        let Value::Tensor(upsampled) = out else {
            panic!("expected tensor");
        };
        assert_eq!(upsampled.shape, vec![1, 6]);
        assert_eq!(
            upsampled.materialize_f64(),
            vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
        );
    }

    #[test]
    fn upsample_column_vector_honors_phase() {
        let out = call_upsample(vec![
            tensor(vec![1.0, 2.0, 3.0], vec![3, 1]),
            Value::Num(3.0),
            Value::Num(1.0),
        ]);
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![9, 1]);
        assert_eq!(
            tensor.materialize_f64(),
            vec![0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0]
        );
    }

    #[test]
    fn upsample_gpu_uses_provider_scatter_and_preserves_residency() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &input.materialize_f64(),
                    shape: &input.shape,
                })
                .expect("upload input");
            provider.reset_telemetry();

            let out = call_upsample(vec![
                Value::GpuTensor(handle.clone()),
                Value::Num(2.0),
                Value::Num(1.0),
            ]);
            let Value::GpuTensor(out_handle) = out else {
                panic!("expected resident gpu tensor");
            };
            assert_eq!(out_handle.shape, vec![4, 2]);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let gathered =
                test_support::gather(Value::GpuTensor(out_handle)).expect("gather output");
            assert_eq!(gathered.shape, vec![4, 2]);
            assert_eq!(
                gathered.materialize_f64(),
                vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0]
            );
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn upsample_gpu_preserves_poisoned_uint64_storage_without_host_download() {
        test_support::with_test_provider(|provider| {
            let input =
                Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]), vec![1, 2])
                    .expect("native uint64 input");
            let handle = gpu_helpers::upload_tensor(provider, &input)
                .expect("upload native uint64 input without mirror materialization");
            provider.reset_telemetry();

            let out = call_upsample(vec![Value::GpuTensor(handle.clone()), Value::Num(2.0)]);
            let Value::GpuTensor(out_handle) = out else {
                panic!("expected resident native integer gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&out_handle),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let gathered = test_support::gather(Value::GpuTensor(out_handle.clone()))
                .expect("gather native integer output");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![1_u64 << 63, 0, u64::MAX, 0]))
            );
            provider.free(&handle).expect("free input");
            provider.free(&out_handle).expect("free output");
        });
    }

    #[test]
    fn host_sample_rate_preserves_exact_uint64_storage() {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .unwrap();
        let Value::Tensor(upsampled) = call_upsample(vec![Value::Tensor(input), Value::Num(2.0)])
        else {
            panic!("expected integer upsample output");
        };
        assert_eq!(
            upsampled.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                9_007_199_254_740_993,
                0,
                u64::MAX,
                0
            ]))
        );
        let Value::Tensor(downsampled) =
            call_downsample(vec![Value::Tensor(upsampled), Value::Num(2.0)])
        else {
            panic!("expected integer downsample output");
        };
        assert_eq!(
            downsampled.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
        );
    }

    #[test]
    fn downsample_row_vector_keeps_every_nth_sample() {
        let out = call_downsample(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]),
            Value::Num(2.0),
        ]);
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![1, 3]);
        assert_eq!(tensor.materialize_f64(), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn downsample_phase_offsets_first_kept_sample() {
        let out = call_downsample(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]),
            Value::Num(2.0),
            Value::Num(1.0),
        ]);
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![1, 2]);
        assert_eq!(tensor.materialize_f64(), vec![2.0, 4.0]);
    }

    #[test]
    fn downsample_preserves_all_integer_classes_exactly() {
        macro_rules! check {
            ($variant:ident, $values:expr, $expected:expr) => {{
                let input = Value::Tensor(
                    Tensor::new_integer(IntegerStorage::$variant($values), vec![1, 5]).unwrap(),
                );
                let output = call_downsample(vec![input, Value::Num(2.0), Value::Num(1.0)]);
                let Value::Tensor(output) = output else {
                    panic!("expected integer tensor")
                };
                assert_eq!(
                    output.integer_storage(),
                    Some(&IntegerStorage::$variant($expected))
                );
                assert_eq!(output.shape, vec![1, 2]);
            }};
        }
        check!(I8, vec![-1, 2, -3, 4, -5], vec![2, 4]);
        check!(I16, vec![-1, 2, -3, 4, -5], vec![2, 4]);
        check!(I32, vec![-1, 2, -3, 4, -5], vec![2, 4]);
        check!(I64, vec![-1, 2, -3, 4, -5], vec![2, 4]);
        check!(U8, vec![1, 2, 3, 4, 5], vec![2, 4]);
        check!(U16, vec![1, 2, 3, 4, 5], vec![2, 4]);
        check!(U32, vec![1, 2, 3, 4, 5], vec![2, 4]);
        check!(U64, vec![1, 2, 3, u64::MAX, 5], vec![2, u64::MAX]);
    }

    #[test]
    fn downsample_typed_controls_are_separately_extension_gated() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let factor_error = block_on(downsample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Int(runmat_builtins::IntValue::U8(2)),
            vec![],
        ))
        .expect_err("integer factor extension");
        assert_eq!(
            factor_error.identifier(),
            DOWNSAMPLE_INTEGER_FACTOR_EXTENSION.error_identifier
        );
        let phase_error = block_on(downsample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(2.0),
            vec![Value::Int(runmat_builtins::IntValue::U8(1))],
        ))
        .expect_err("integer phase extension");
        assert_eq!(
            phase_error.identifier(),
            DOWNSAMPLE_INTEGER_PHASE_EXTENSION.error_identifier
        );
        drop(strict);
    }

    #[test]
    fn downsample_nd_input_is_extension_gated() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(downsample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 1, 2]),
            Value::Num(2.0),
            Vec::new(),
        ))
        .expect_err("N-D input extension gate");
        assert_eq!(
            error.identifier(),
            DOWNSAMPLE_ND_INPUT_EXTENSION.error_identifier
        );
        drop(strict);

        let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let output = block_on(downsample_builtin(
            tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]),
            Value::Num(2.0),
            Vec::new(),
        ))
        .expect("N-D input in RunMat mode");
        let Value::Tensor(output) = output else {
            panic!("expected tensor")
        };
        assert_eq!(output.shape, vec![1, 1, 2]);
        assert_eq!(output.materialize_f64(), vec![1.0, 3.0]);
        drop(extensions);
    }

    #[test]
    fn downsample_resident_integer_preserves_owner_class_and_values() {
        use runmat_accelerate_api::{
            HostIntegerDataView, HostIntegerTensorView, IntegerElementType,
        };

        test_support::with_test_provider(|provider| {
            let values = [1u64, 2, 3, u64::MAX, 5];
            let input = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&values),
                    shape: &[1, 5],
                })
                .expect("integer upload");
            let output = call_downsample(vec![
                Value::GpuTensor(input.clone()),
                Value::Num(2.0),
                Value::Num(1.0),
            ]);
            let Value::GpuTensor(output) = output else {
                panic!("expected resident integer output")
            };
            assert_eq!(output.device_id, input.device_id);
            assert_eq!(output.shape, vec![1, 2]);
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&output),
                Some(IntegerElementType::U64)
            );
            assert!(runmat_accelerate_api::provider_for_handle(&output)
                .is_some_and(|owner| std::ptr::eq(owner, provider)));
            let gathered = block_on(provider.download_integer(&output)).expect("integer download");
            assert_eq!(
                gathered.data,
                runmat_accelerate_api::HostIntegerDataOwned::U64(vec![2, u64::MAX])
            );
            let _ = provider.free(&input);
            let _ = provider.free(&output);
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn downsample_wgpu_integer_data_and_typed_controls_stay_exact_and_resident() {
        use runmat_accelerate_api::{
            HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView, IntegerElementType,
        };

        let _accel_guard = test_support::accel_test_lock();
        match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) {
            Ok(_) => {}
            Err(error) if error.to_string() == "wgpu: no compatible adapter found" => return,
            Err(error) => panic!("register wgpu provider failed: {error:?}"),
        }
        let provider = runmat_accelerate_api::provider().expect("WGPU provider");
        let values = [
            1_u64,
            9_007_199_254_740_993,
            3,
            u64::MAX,
            5,
            9_007_199_254_740_995,
        ];
        let input = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&values),
                shape: &[1, 6],
            })
            .expect("upload WGPU uint64 input");

        let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let output = call_downsample(vec![
            Value::GpuTensor(input.clone()),
            Value::Int(runmat_builtins::IntValue::U8(2)),
            Value::Int(runmat_builtins::IntValue::U8(1)),
        ]);
        drop(extensions);
        let Value::GpuTensor(output) = output else {
            panic!("expected resident WGPU integer output")
        };

        assert_eq!(output.device_id, input.device_id);
        assert_eq!(output.shape, vec![1, 3]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(IntegerElementType::U64)
        );
        assert!(runmat_accelerate_api::provider_for_handle(&output)
            .is_some_and(|owner| std::ptr::eq(owner, provider)));
        let downloaded =
            block_on(provider.download_integer(&output)).expect("download WGPU uint64 output");
        assert_eq!(downloaded.shape, vec![1, 3]);
        assert_eq!(
            downloaded.data,
            HostIntegerDataOwned::U64(
                vec![9_007_199_254_740_993, u64::MAX, 9_007_199_254_740_995,]
            )
        );

        provider.free(&input).expect("free WGPU input");
        provider.free(&output).expect("free WGPU output");
    }

    #[test]
    fn downsample_unknown_input_device_rejects_before_provider_work() {
        let _guard = test_support::accel_test_lock();
        let provider = Box::leak(Box::new(HostileDownsampleProvider::new()));
        unsafe {
            runmat_accelerate_api::register_provider(provider);
        }

        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0],
                shape: &[1, 4],
            })
            .expect("input upload");
        let hostile = GpuTensorHandle {
            shape: input.shape.clone(),
            device_id: input.device_id.wrapping_add(10_000),
            buffer_id: input.buffer_id,
        };
        let frees_before = provider.frees.load(Ordering::Relaxed);

        let error = block_on(downsample_builtin(
            Value::GpuTensor(hostile),
            Value::Num(2.0),
            Vec::new(),
        ))
        .expect_err("unknown input device must reject before provider work");
        assert_eq!(error.identifier(), SAMPLE_ERROR_GATHER_FAILED.identifier);
        assert!(error.message().contains("no proven owner"));
        assert_eq!(provider.gathers.load(Ordering::Relaxed), 0);
        assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
        assert_eq!(provider.frees.load(Ordering::Relaxed), frees_before);
        assert!(provider
            .buffers
            .lock()
            .unwrap()
            .contains_key(&input.buffer_id));

        provider.free(&input).expect("free live input");
        assert!(provider.buffers.lock().unwrap().is_empty());
    }

    #[test]
    fn downsample_rejects_hostile_provider_results_and_falls_back_only_on_hook_error() {
        let _guard = test_support::accel_test_lock();
        let provider = Box::leak(Box::new(HostileDownsampleProvider::new()));
        unsafe {
            runmat_accelerate_api::register_provider(provider);
        }

        for mode in 0..6_u8 {
            provider.mode.store(mode, Ordering::Relaxed);
            let input = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, 3.0, 4.0],
                    shape: &[1, 4],
                })
                .expect("input upload");
            let frees_before = provider.frees.load(Ordering::Relaxed);
            let error = downsample_gpu(&input, 2, 0, DOWNSAMPLE_NAME)
                .expect_err("malformed provider result must be an explicit error");
            assert_eq!(error.identifier(), SAMPLE_ERROR_GATHER_FAILED.identifier);

            if mode == 1 {
                assert_eq!(
                    provider.frees.load(Ordering::Relaxed),
                    frees_before,
                    "unknown-owner output must not be freed through the input owner"
                );
                let unknown = GpuTensorHandle {
                    shape: vec![1, 2],
                    device_id: provider.device_id.wrapping_add(10_000),
                    buffer_id: provider.last_output.load(Ordering::Relaxed),
                };
                provider.free(&unknown).expect("test cleanup");
            } else {
                assert_eq!(
                    provider.frees.load(Ordering::Relaxed),
                    frees_before + 1,
                    "known-owner malformed output must be freed exactly once"
                );
            }
            provider.free(&input).expect("free input");
            assert!(provider.buffers.lock().unwrap().is_empty());
        }

        provider.mode.store(7, Ordering::Relaxed);
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0],
                shape: &[1, 4],
            })
            .expect("alias input upload");
        let frees_before = provider.frees.load(Ordering::Relaxed);
        let error = downsample_gpu(&input, 2, 0, DOWNSAMPLE_NAME)
            .expect_err("provider output alias must be an explicit error");
        assert_eq!(error.identifier(), SAMPLE_ERROR_GATHER_FAILED.identifier);
        assert_eq!(
            provider.frees.load(Ordering::Relaxed),
            frees_before,
            "aliased input must remain caller-owned"
        );
        assert!(provider
            .buffers
            .lock()
            .unwrap()
            .contains_key(&input.buffer_id));
        provider.free(&input).expect("free aliased input");
        assert!(provider.buffers.lock().unwrap().is_empty());

        provider.mode.store(6, Ordering::Relaxed);
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0],
                shape: &[1, 4],
            })
            .expect("input upload");
        let output = block_on(downsample_builtin(
            Value::GpuTensor(input.clone()),
            Value::Num(2.0),
            Vec::new(),
        ))
        .expect("unsupported gather hook uses the documented host fallback");
        let Value::Tensor(output) = output else {
            panic!("fallback must return a host tensor")
        };
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(output.materialize_f64(), vec![1.0, 3.0]);
        provider.free(&input).expect("free fallback input");
        assert!(provider.buffers.lock().unwrap().is_empty());
    }

    #[test]
    fn downsample_gpu_uses_provider_linear_gather_and_preserves_residency() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &input.materialize_f64(),
                    shape: &input.shape,
                })
                .expect("upload input");
            provider.reset_telemetry();

            let out = call_downsample(vec![
                Value::GpuTensor(handle.clone()),
                Value::Num(2.0),
                Value::Num(1.0),
            ]);
            let Value::GpuTensor(out_handle) = out else {
                panic!("expected resident gpu tensor");
            };
            assert_eq!(out_handle.shape, vec![1, 2]);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let gathered =
                test_support::gather(Value::GpuTensor(out_handle)).expect("gather output");
            assert_eq!(gathered.shape, vec![1, 2]);
            assert_eq!(gathered.materialize_f64(), vec![2.0, 4.0]);
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn upsample_complex_gpu_stays_resident_and_preserves_interleaved_storage() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload");
            provider.reset_telemetry();

            let out = call_upsample(vec![
                Value::GpuTensor(handle.clone()),
                Value::Num(2.0),
                Value::Num(1.0),
            ]);
            let Value::GpuTensor(out_handle) = out else {
                panic!("expected resident gpu tensor");
            };
            assert_eq!(out_handle.shape, vec![1, 4]);
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out_handle),
                GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let host = block_on(provider.download(&out_handle)).expect("download output");
            assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
            assert_eq!(host.shape, vec![1, 4]);
            assert_eq!(host.data, vec![0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
            let _ = provider.free(&handle);
            let _ = provider.free(&out_handle);
        });
    }

    #[test]
    fn downsample_complex_gpu_stays_resident_and_gathers_logical_elements() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(
                vec![(1.0, -1.0), (2.0, -2.0), (3.0, -3.0), (4.0, -4.0)],
                vec![1, 4],
            )
            .unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload");
            provider.reset_telemetry();

            let out = call_downsample(vec![
                Value::GpuTensor(handle.clone()),
                Value::Num(2.0),
                Value::Num(1.0),
            ]);
            let Value::GpuTensor(out_handle) = out else {
                panic!("expected resident gpu tensor");
            };
            assert_eq!(out_handle.shape, vec![1, 2]);
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out_handle),
                GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let host = block_on(provider.download(&out_handle)).expect("download output");
            assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
            assert_eq!(host.shape, vec![1, 2]);
            assert_eq!(host.data, vec![2.0, -2.0, 4.0, -4.0]);
            let _ = provider.free(&handle);
            let _ = provider.free(&out_handle);
        });
    }

    #[test]
    fn sample_rate_operates_down_matrix_columns() {
        let out = call_upsample(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            Value::Num(2.0),
        ]);
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![4, 2]);
        assert_eq!(
            tensor.materialize_f64(),
            vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]
        );

        let down = call_downsample(vec![Value::Tensor(tensor), Value::Num(2.0)]);
        let Value::Tensor(tensor) = down else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn sample_rate_operates_along_first_non_singleton_nd_dimension() {
        let out = call_upsample(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]),
            Value::Num(2.0),
        ]);
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![1, 4, 2]);
        assert_eq!(
            tensor.materialize_f64(),
            vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]
        );
    }

    #[test]
    fn sample_rate_preserves_empty_shape() {
        let out = call_upsample(vec![tensor(Vec::new(), vec![1, 0]), Value::Num(3.0)]);
        let Value::Tensor(array) = out else {
            panic!("expected tensor");
        };
        assert_eq!(array.shape, vec![1, 0]);
        assert!(array.materialize_f64().is_empty());

        let out = call_downsample(vec![tensor(Vec::new(), vec![0, 1]), Value::Num(2.0)]);
        let Value::Tensor(array) = out else {
            panic!("expected tensor");
        };
        assert_eq!(array.shape, vec![0, 1]);
        assert!(array.materialize_f64().is_empty());
    }

    #[test]
    fn sample_rate_preserves_complex_values() {
        let input = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap(),
        );
        let out = call_upsample(vec![input, Value::Num(2.0)]);
        let Value::ComplexTensor(tensor) = out else {
            panic!("expected complex tensor");
        };
        assert_eq!(tensor.shape, vec![1, 4]);
        assert_eq!(
            tensor.materialize_f64(),
            vec![(1.0, 2.0), (0.0, 0.0), (3.0, 4.0), (0.0, 0.0)]
        );
    }

    #[test]
    fn sample_rate_rejects_invalid_factor_and_phase() {
        let err = block_on(upsample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(0.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_FACTOR.identifier);

        let err = block_on(upsample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(2.0000004),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_FACTOR.identifier);

        let err = block_on(downsample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(2.0),
            vec![Value::Num(2.0)],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_PHASE.identifier);

        let err = block_on(downsample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(2.0),
            vec![Value::Num(1.0000004)],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_PHASE.identifier);
    }

    #[test]
    fn resample_custom_unit_filter_matches_zero_insert_and_drop() {
        let out = call_resample(vec![
            tensor(vec![1.0, 2.0, 3.0], vec![1, 3]),
            Value::Num(2.0),
            Value::Num(1.0),
            tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
        ]);
        let Value::Tensor(upsampled) = out else {
            panic!("expected tensor");
        };
        assert_eq!(upsampled.shape, vec![1, 6]);
        assert_eq!(
            upsampled.materialize_f64(),
            vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
        );

        let down = call_resample(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]),
            Value::Num(1.0),
            Value::Num(2.0),
            tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
        ]);
        let Value::Tensor(downsampled) = down else {
            panic!("expected tensor");
        };
        assert_eq!(downsampled.shape, vec![1, 3]);
        assert_eq!(downsampled.materialize_f64(), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn resample_rejects_integer_signal_storage() {
        let input = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3]).unwrap();
        let error = block_on(resample_builtin(
            Value::Tensor(input),
            Value::Num(2.0),
            Value::Num(1.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), SAMPLE_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn resample_filter_vector_reads_typed_integer_storage_exactly() {
        let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
        let filter = Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 0]), vec![1, 3]).unwrap();

        let parsed = block_on(parse_filter_vector(Value::Tensor(filter))).unwrap();
        let values = block_on(parsed.host_values()).unwrap();
        assert_eq!(values, vec![0.0, 1.0, 0.0]);

        let out = call_resample(vec![
            tensor(vec![1.0, 2.0, 3.0], vec![1, 3]),
            Value::Num(2.0),
            Value::Num(1.0),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 0]), vec![1, 3]).unwrap(),
            ),
        ]);
        let Value::Tensor(upsampled) = out else {
            panic!("expected tensor");
        };
        assert_eq!(upsampled.shape, vec![1, 6]);
        assert_eq!(
            upsampled.materialize_f64(),
            vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
        );
    }

    #[test]
    fn resample_scalar_integer_option_reads_typed_integer_storage_without_mirror() {
        let scalar =
            Tensor::new_integer(IntegerStorage::U16(vec![12]), vec![1, 1]).expect("scalar");
        let vector =
            Tensor::new_integer(IntegerStorage::U16(vec![12, 14]), vec![1, 2]).expect("vector");

        assert_eq!(
            block_on(scalar_integer_option(&Value::Tensor(scalar))).expect("scalar"),
            Some(12)
        );
        assert_eq!(
            block_on(scalar_integer_option(&Value::Tensor(vector))).expect("vector"),
            None
        );
    }

    #[test]
    fn sample_rate_integer_parsers_read_typed_integer_storage_exactly() {
        let wide = exact_platform_wide_integer();
        let factor =
            Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("factor");
        let phase =
            Tensor::new_integer(IntegerStorage::U64(vec![wide - 1]), vec![1, 1]).expect("phase");

        assert_eq!(
            block_on(parse_factor(&Value::Tensor(factor), UPSAMPLE_NAME)).expect("factor"),
            wide as usize
        );
        assert_eq!(
            block_on(parse_phase(
                &Value::Tensor(phase),
                wide as usize,
                UPSAMPLE_NAME
            ))
            .expect("phase"),
            (wide - 1) as usize
        );
    }

    #[test]
    fn sample_rate_integer_parsers_reject_unrepresentable_double_boundary() {
        let err = parse_nonnegative_integer(
            first_unrepresentable_usize_double(),
            UPSAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_FACTOR,
        )
        .expect_err("unrepresentable integer should fail");

        assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_FACTOR.identifier);
    }

    #[test]
    fn resample_gpu_composes_scatter_filter_and_gather_resident_output() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &input.materialize_f64(),
                    shape: &input.shape,
                })
                .expect("upload input");
            provider.reset_telemetry();

            let out = call_resample(vec![
                Value::GpuTensor(handle.clone()),
                Value::Num(2.0),
                Value::Num(1.0),
                tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
            ]);
            let Value::GpuTensor(out_handle) = out else {
                panic!("expected resident gpu tensor");
            };
            assert_eq!(out_handle.shape, vec![1, 6]);

            let gathered =
                test_support::gather(Value::GpuTensor(out_handle)).expect("gather output");
            assert_eq!(gathered.shape, vec![1, 6]);
            assert_eq!(
                gathered.materialize_f64(),
                vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
            );
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn resample_gpu_accepts_resident_custom_filter_without_download() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &input.materialize_f64(),
                    shape: &input.shape,
                })
                .expect("upload input");
            let filter = [0.0, 1.0, 0.0];
            let filter_shape = [1usize, 3usize];
            let filter_handle = provider
                .upload(&HostTensorView {
                    data: &filter,
                    shape: &filter_shape,
                })
                .expect("upload filter");
            provider.reset_telemetry();

            let out = call_resample(vec![
                Value::GpuTensor(handle.clone()),
                Value::Num(2.0),
                Value::Num(1.0),
                Value::GpuTensor(filter_handle.clone()),
            ]);
            let Value::GpuTensor(out_handle) = out else {
                panic!("expected resident gpu tensor");
            };
            assert_eq!(out_handle.shape, vec![1, 6]);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let gathered =
                test_support::gather(Value::GpuTensor(out_handle.clone())).expect("gather output");
            assert_eq!(gathered.shape, vec![1, 6]);
            assert_eq!(
                gathered.materialize_f64(),
                vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
            );
            let _ = provider.free(&handle);
            let _ = provider.free(&filter_handle);
            let _ = provider.free(&out_handle);
        });
    }

    #[test]
    fn resample_complex_gpu_stays_resident_through_filter_pipeline() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload");
            provider.reset_telemetry();

            let out = call_resample(vec![
                Value::GpuTensor(handle.clone()),
                Value::Num(2.0),
                Value::Num(1.0),
                tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
            ]);
            let Value::GpuTensor(out_handle) = out else {
                panic!("expected resident gpu tensor");
            };
            assert_eq!(out_handle.shape, vec![1, 4]);
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out_handle),
                GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let host = block_on(provider.download(&out_handle)).expect("download output");
            assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
            assert_eq!(host.shape, vec![1, 4]);
            assert_eq!(host.data, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
            let _ = provider.free(&handle);
            let _ = provider.free(&out_handle);
        });
    }

    #[test]
    fn resample_gpu_two_outputs_return_resident_signal_and_host_filter() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &input.materialize_f64(),
                    shape: &input.shape,
                })
                .expect("upload input");

            let _guard = crate::output_count::push_output_count(Some(2));
            let out = block_on(resample_builtin(
                Value::GpuTensor(handle.clone()),
                Value::Num(1.0),
                Value::Num(2.0),
                vec![tensor(vec![0.0, 1.0, 0.0], vec![1, 3])],
            ))
            .expect("gpu resample");
            let Value::OutputList(outputs) = out else {
                panic!("expected output list");
            };
            assert_eq!(outputs.len(), 2);
            let Value::GpuTensor(signal_handle) = &outputs[0] else {
                panic!("expected resident signal output");
            };
            assert_eq!(signal_handle.shape, vec![1, 3]);
            let gathered = test_support::gather(outputs[0].clone()).expect("gather signal output");
            assert_eq!(gathered.materialize_f64(), vec![1.0, 3.0, 5.0]);
            let Value::Tensor(filter) = &outputs[1] else {
                panic!("expected host filter tensor");
            };
            assert_eq!(filter.materialize_f64(), vec![0.0, 1.0, 0.0]);
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn resample_gpu_short_filter_high_decimation_stays_resident() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new((1..=10).map(f64::from).collect(), vec![1, 10]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &input.materialize_f64(),
                    shape: &input.shape,
                })
                .expect("upload input");

            let out = call_resample(vec![
                Value::GpuTensor(handle.clone()),
                Value::Num(1.0),
                Value::Num(10.0),
                tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
            ]);
            let Value::GpuTensor(out_handle) = out else {
                panic!("expected resident gpu tensor");
            };
            assert_eq!(out_handle.shape, vec![1, 1]);
            let gathered =
                test_support::gather(Value::GpuTensor(out_handle)).expect("gather output");
            assert_eq!(gathered.materialize_f64(), vec![1.0]);
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn resample_dimension_name_value_operates_on_requested_axis() {
        let out = call_resample(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            Value::Num(2.0),
            Value::Num(1.0),
            tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
            Value::CharArray(runmat_builtins::CharArray::new_row("Dimension")),
            Value::Num(2.0),
        ]);
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 4]);
        assert_eq!(
            tensor.materialize_f64(),
            vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]
        );
    }

    #[test]
    fn resample_preserves_complex_domain() {
        let input = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap(),
        );
        let out = call_resample(vec![
            input,
            Value::Num(2.0),
            Value::Num(1.0),
            tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
        ]);
        let Value::ComplexTensor(tensor) = out else {
            panic!("expected complex tensor");
        };
        assert_eq!(tensor.shape, vec![1, 4]);
        assert_eq!(
            tensor.materialize_f64(),
            vec![(1.0, 2.0), (0.0, 0.0), (3.0, 4.0), (0.0, 0.0)]
        );
    }

    #[test]
    fn resample_two_outputs_return_signal_and_filter() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = block_on(resample_builtin(
            tensor(vec![1.0, 2.0, 3.0], vec![3, 1]),
            Value::Num(3.0),
            Value::Num(2.0),
            vec![Value::Num(1.0), Value::Num(0.0)],
        ))
        .unwrap();
        let Value::OutputList(values) = result else {
            panic!("expected output list");
        };
        assert_eq!(values.len(), 2);
        let Value::Tensor(signal) = &values[0] else {
            panic!("expected signal tensor");
        };
        let Value::Tensor(filter) = &values[1] else {
            panic!("expected filter tensor");
        };
        assert_eq!(signal.shape, vec![5, 1]);
        assert_eq!(filter.shape, vec![1, 7]);
        let sum = filter.materialize_f64().iter().sum::<f64>();
        assert!((sum - 3.0).abs() < 1e-10);
    }

    #[test]
    fn resample_default_filter_has_matlab_compatible_length() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = block_on(resample_builtin(
            tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]),
            Value::Num(4.0),
            Value::Num(2.0),
            Vec::new(),
        ))
        .unwrap();
        let Value::OutputList(values) = result else {
            panic!("expected output list");
        };
        let Value::Tensor(signal) = &values[0] else {
            panic!("expected signal tensor");
        };
        let Value::Tensor(filter) = &values[1] else {
            panic!("expected filter tensor");
        };
        assert_eq!(signal.shape, vec![1, 8]);
        assert_eq!(filter.shape, vec![1, 41]);
        let sum = filter.materialize_f64().iter().sum::<f64>();
        assert!((sum - 2.0).abs() < 1e-10);
    }

    #[test]
    fn resample_filter_design_rejects_huge_allocation() {
        let err = design_resample_filter(1usize << (usize::BITS - 3), 1, 1, 5.0).unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_SIZE_OVERFLOW.identifier);
    }

    #[test]
    fn resample_rejects_invalid_options_and_output_count() {
        let err = block_on(resample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(2.0),
            Value::Num(0.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_FACTOR.identifier);

        let err = block_on(resample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(2.0),
            Value::Num(1.0),
            vec![Value::CharArray(runmat_builtins::CharArray::new_row(
                "Dimension",
            ))],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_OPTION.identifier);

        let _guard = crate::output_count::push_output_count(Some(3));
        let err = block_on(resample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(2.0),
            Value::Num(1.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_OUTPUT_COUNT.identifier);
    }
}
