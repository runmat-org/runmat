//! Probability distribution compatibility helpers.

use runmat_accelerate_api::{GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, NumericScalar, Tensor, Value};

use crate::builtins::common::{broadcast, gpu_helpers, tensor};
use crate::builtins::math::elementwise::error_functions::erfcinv::erfcinv_scalar;
use crate::builtins::stats::summary::distribution_math;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const SQRT_2: f64 = std::f64::consts::SQRT_2;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

const OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Distribution function value.",
}];

const INPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Evaluation point.",
};

const INPUT_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Probability value.",
};

const INPUT_DIST: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Distribution name.",
};

const INPUT_MU: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mu",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("0"),
    description: "Mean parameter.",
};

const INPUT_SIGMA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sigma",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "Standard deviation parameter.",
};

const INPUT_NU: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nu",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Degrees of freedom parameter.",
};

const INPUT_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of trials.",
};

const INPUT_PROB: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "prob",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Success probability parameter.",
};

const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "a",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "Scale parameter.",
};

const INPUT_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "Shape parameter.",
};

const INPUT_SHAPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "shape",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Shape parameter.",
};

const INPUT_SCALE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "scale",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scale parameter.",
};

const INPUT_LAMBDA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "lambda",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Rate or mean parameter.",
};

const INPUT_LOWER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "a",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Lower endpoint.",
};

const INPUT_UPPER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Upper endpoint.",
};

const INPUT_DF1: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "v1",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numerator degrees of freedom.",
};

const INPUT_DF2: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "v2",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Denominator degrees of freedom.",
};

const INPUT_ALPHA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "a",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "First shape parameter.",
};

const INPUT_BETA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second shape parameter.",
};

const INPUT_MEAN_REQUIRED: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mu",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mean parameter.",
};

const INPUTS_X: [BuiltinParamDescriptor; 1] = [INPUT_X];
const INPUTS_X_MU: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_MU];
const INPUTS_X_MU_SIGMA: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_MU, INPUT_SIGMA];
const INPUTS_P: [BuiltinParamDescriptor; 1] = [INPUT_P];
const INPUTS_DIST_P: [BuiltinParamDescriptor; 2] = [INPUT_DIST, INPUT_P];
const INPUTS_DIST_P_PARAM: [BuiltinParamDescriptor; 3] = [INPUT_DIST, INPUT_P, INPUT_NU];
const INPUTS_DIST_P_MU_SIGMA: [BuiltinParamDescriptor; 4] =
    [INPUT_DIST, INPUT_P, INPUT_MU, INPUT_SIGMA];
const INPUTS_DIST_P_MU: [BuiltinParamDescriptor; 3] = [INPUT_DIST, INPUT_P, INPUT_MEAN_REQUIRED];
const INPUTS_DIST_P_A_B: [BuiltinParamDescriptor; 4] = [INPUT_DIST, INPUT_P, INPUT_A, INPUT_B];
const INPUTS_DIST_P_SHAPE_SCALE: [BuiltinParamDescriptor; 4] =
    [INPUT_DIST, INPUT_P, INPUT_SHAPE, INPUT_SCALE];
const INPUTS_DIST_P_N_PROB: [BuiltinParamDescriptor; 4] =
    [INPUT_DIST, INPUT_P, INPUT_N, INPUT_PROB];
const INPUTS_DIST_P_LAMBDA: [BuiltinParamDescriptor; 3] = [INPUT_DIST, INPUT_P, INPUT_LAMBDA];
const INPUTS_DIST_P_BOUNDS: [BuiltinParamDescriptor; 4] =
    [INPUT_DIST, INPUT_P, INPUT_LOWER, INPUT_UPPER];
const INPUTS_DIST_P_DF: [BuiltinParamDescriptor; 4] = [INPUT_DIST, INPUT_P, INPUT_DF1, INPUT_DF2];
const INPUTS_DIST_P_ALPHA_BETA: [BuiltinParamDescriptor; 4] =
    [INPUT_DIST, INPUT_P, INPUT_ALPHA, INPUT_BETA];
const INPUTS_P_MU: [BuiltinParamDescriptor; 2] = [INPUT_P, INPUT_MU];
const INPUTS_P_MU_SIGMA: [BuiltinParamDescriptor; 3] = [INPUT_P, INPUT_MU, INPUT_SIGMA];
const INPUTS_X_NU: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_NU];
const INPUT_TAIL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tail",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"upper\""),
    description: "Upper-tail selector (`\"upper\"`).",
};
const INPUTS_X_NU_UPPER: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_NU, INPUT_TAIL];
const INPUTS_P_NU: [BuiltinParamDescriptor; 2] = [INPUT_P, INPUT_NU];
const INPUTS_X_N_P: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_N, INPUT_PROB];
const INPUTS_P_A_B: [BuiltinParamDescriptor; 3] = [INPUT_P, INPUT_A, INPUT_B];

const NORMAL_SIGNATURES_X: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "y = normpdf(x)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = normpdf(x, mu)",
        inputs: &INPUTS_X_MU,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = normpdf(x, mu, sigma)",
        inputs: &INPUTS_X_MU_SIGMA,
        outputs: &OUTPUT_Y,
    },
];

const NORMAL_CDF_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "p = normcdf(x)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "p = normcdf(x, mu)",
        inputs: &INPUTS_X_MU,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "p = normcdf(x, mu, sigma)",
        inputs: &INPUTS_X_MU_SIGMA,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "p = normcdf(x, mu, sigma, \"upper\")",
        inputs: &INPUTS_X_MU_SIGMA,
        outputs: &OUTPUT_Y,
    },
];

const NORMAL_INV_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "x = norminv(p)",
        inputs: &INPUTS_P,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = norminv(p, mu)",
        inputs: &INPUTS_P_MU,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = norminv(p, mu, sigma)",
        inputs: &INPUTS_P_MU_SIGMA,
        outputs: &OUTPUT_Y,
    },
];

const T_PDF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "y = tpdf(x, nu)",
    inputs: &INPUTS_X_NU,
    outputs: &OUTPUT_Y,
}];

const T_CDF_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "p = tcdf(x, nu)",
        inputs: &INPUTS_X_NU,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "p = tcdf(x, nu, \"upper\")",
        inputs: &INPUTS_X_NU,
        outputs: &OUTPUT_Y,
    },
];

const T_INV_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "x = tinv(p, nu)",
    inputs: &INPUTS_P_NU,
    outputs: &OUTPUT_Y,
}];

const BINOCDF_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "y = binocdf(x, n, p)",
        inputs: &INPUTS_X_N_P,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = binocdf(x, n, p, \"upper\")",
        inputs: &INPUTS_X_N_P,
        outputs: &OUTPUT_Y,
    },
];

const CHI2CDF_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "p = chi2cdf(x, nu)",
        inputs: &INPUTS_X_NU,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "p = chi2cdf(x, nu, \"upper\")",
        inputs: &INPUTS_X_NU_UPPER,
        outputs: &OUTPUT_Y,
    },
];

const WBLINV_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "x = wblinv(p)",
        inputs: &INPUTS_P,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = wblinv(p, a, b)",
        inputs: &INPUTS_P_A_B,
        outputs: &OUTPUT_Y,
    },
];

const ICDF_SIGNATURES: [BuiltinSignatureDescriptor; 15] = [
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Normal\", p)",
        inputs: &INPUTS_DIST_P,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Normal\", p, mu, sigma)",
        inputs: &INPUTS_DIST_P_MU_SIGMA,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"t\", p, nu)",
        inputs: &INPUTS_DIST_P_PARAM,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Weibull\", p, a, b)",
        inputs: &INPUTS_DIST_P_A_B,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Chi-square\", p, nu)",
        inputs: &INPUTS_DIST_P_PARAM,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Binomial\", p, n, prob)",
        inputs: &INPUTS_DIST_P_N_PROB,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Gamma\", p, shape, scale)",
        inputs: &INPUTS_DIST_P_SHAPE_SCALE,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Exponential\", p, mu)",
        inputs: &INPUTS_DIST_P_MU,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Poisson\", p, lambda)",
        inputs: &INPUTS_DIST_P_LAMBDA,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Uniform\", p)",
        inputs: &INPUTS_DIST_P,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Uniform\", p, a, b)",
        inputs: &INPUTS_DIST_P_BOUNDS,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Lognormal\", p, mu, sigma)",
        inputs: &INPUTS_DIST_P_MU_SIGMA,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Beta\", p, a, b)",
        inputs: &INPUTS_DIST_P_ALPHA_BETA,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"F\", p, v1, v2)",
        inputs: &INPUTS_DIST_P_DF,
        outputs: &OUTPUT_Y,
    },
    crate::builtins::stats::summary::fitdist::ICDF_OBJECT_SIGNATURE,
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NORMAL.INVALID_ARGUMENT",
    identifier: None,
    when: "Inputs are nonnumeric, sizes are incompatible, or too many arguments are supplied.",
    message: "normal distribution: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NORMAL.INTERNAL",
    identifier: None,
    when: "Internal tensor conversion or allocation fails.",
    message: "normal distribution: internal error",
};

macro_rules! normal_descriptor {
    ($name:literal, $signatures:expr) => {
        const ERRORS: [BuiltinErrorDescriptor; 2] = [
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INVALID_ARGUMENT"),
                identifier: Some(concat!("RunMat:", $name, ":InvalidArgument")),
                when: ERROR_INVALID_ARGUMENT.when,
                message: ERROR_INVALID_ARGUMENT.message,
            },
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INTERNAL"),
                identifier: Some(concat!("RunMat:", $name, ":Internal")),
                when: ERROR_INTERNAL.when,
                message: ERROR_INTERNAL.message,
            },
        ];

        pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

fn normal_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn icdf_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let mut shape: Option<Vec<Option<usize>>> = None;
    for arg in args.iter().skip(1) {
        let current = match arg {
            Type::Num | Type::Int | Type::Bool => {
                runmat_builtins::shape_rules::scalar_tensor_shape()
            }
            Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
                shape.clone()
            }
            Type::Tensor { shape: None } | Type::Logical { shape: None } => {
                return Type::Tensor { shape: None };
            }
            Type::Unknown => return Type::Unknown,
            _ => return Type::Unknown,
        };
        shape = Some(match shape {
            Some(previous) => runmat_builtins::shape_rules::broadcast_shapes(&previous, &current),
            None => current,
        });
    }
    shape
        .map(runmat_builtins::shape_rules::numeric_tensor_from_shape)
        .unwrap_or(Type::Unknown)
}

fn normal_error(name: &str, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(name);
    if name != "normal" {
        builder = builder.with_identifier(format!("RunMat:{name}:InvalidArgument"));
    }
    builder.build()
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

fn ensure_exact_integer_boundary(name: &str, tensor: &Tensor, role: &str) -> BuiltinResult<()> {
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
            return Err(normal_error(
                name,
                format!("{name}: integer {role} values must be exactly representable as double"),
            ));
        }
    }
    Ok(())
}

async fn normal_value_to_tensor(name: &str, role: &str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| normal_error(name, format!("{name}: {err}")))?;
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| normal_error(name, format!("{name}: {err}")))?;
    ensure_exact_integer_boundary(name, &tensor, role)?;
    tensor::integer_tensor_to_f64(tensor)
        .map_err(|err| normal_error(name, format!("{name}: {err}")))
}

#[derive(Clone, Copy)]
enum NormalOutputPrecision {
    Double,
    Single,
}

struct NormalOutputPlan {
    precision: NormalOutputPrecision,
    gpu_source: Option<GpuTensorHandle>,
}

impl NormalOutputPlan {
    fn inspect(name: &str, first: &Value, rest: &[Value]) -> BuiltinResult<Self> {
        let precision = if std::iter::once(first)
            .chain(rest.iter())
            .any(is_single_value)
        {
            NormalOutputPrecision::Single
        } else {
            NormalOutputPrecision::Double
        };
        let gpu_source = gpu_helpers::select_resident_output_source(
            std::iter::once(first)
                .chain(rest.iter())
                .filter_map(|value| match value {
                    Value::GpuTensor(handle) => Some(handle.clone()),
                    _ => None,
                }),
            name,
        )?;
        Ok(Self {
            precision,
            gpu_source,
        })
    }

    fn finish(self, name: &str, shape: Vec<usize>, data: Vec<f64>) -> BuiltinResult<Value> {
        let tensor = match self.precision {
            NormalOutputPrecision::Double => Tensor::new(data, shape),
            NormalOutputPrecision::Single => {
                Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
            }
        }
        .map_err(|err| normal_error(name, format!("{name}: {err}")))?;
        let Some(source) = self.gpu_source else {
            return Ok(match self.precision {
                NormalOutputPrecision::Double => tensor::tensor_into_value(tensor),
                NormalOutputPrecision::Single => Value::Tensor(tensor),
            });
        };
        let restored =
            gpu_helpers::restore_class_preserving_value(&source, Value::Tensor(tensor), name)
                .map_err(|err| normal_error(name, format!("{name}: {err}")))?;
        if runmat_accelerate_api::handle_is_explicit(&source)
            && !matches!(restored, Value::GpuTensor(_))
        {
            return Err(normal_error(
                name,
                format!("{name}: provider cannot preserve explicit gpuArray output residency"),
            ));
        }
        Ok(match (self.precision, restored) {
            (NormalOutputPrecision::Double, Value::Tensor(tensor)) => {
                tensor::tensor_into_value(tensor)
            }
            (_, value) => value,
        })
    }
}

macro_rules! normal_integer_surface {
    (
        $builtin:literal,
        $x_id:literal, $x_error:literal,
        $mu_id:literal, $mu_error:literal,
        $sigma_id:literal, $sigma_error:literal,
        $logical_id:literal, $logical_error:literal
    ) => {
        const INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $x_id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: concat!($builtin, " with typed-integer evaluation values is a RunMat extension"),
            error_identifier: Some($x_error),
        };
        const INTEGER_MU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $mu_id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: concat!($builtin, " with typed-integer mean parameters is a RunMat extension"),
            error_identifier: Some($mu_error),
        };
        const INTEGER_SIGMA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $sigma_id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: concat!($builtin, " with typed-integer standard-deviation parameters is a RunMat extension"),
            error_identifier: Some($sigma_error),
        };
        const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $logical_id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: concat!($builtin, " with logical numeric inputs is a RunMat extension"),
            error_identifier: Some($logical_error),
        };
        pub const EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
            INTEGER_X_EXTENSION,
            INTEGER_MU_EXTENSION,
            INTEGER_SIGMA_EXTENSION,
            LOGICAL_INPUT_EXTENSION,
        ];
        const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
            BuiltinIntegerInputCapability {
                name: "x or p",
                classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
                availability: BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: "Typed-integer evaluation values are gated independently and require exact binary64 representation.",
            },
            BuiltinIntegerInputCapability {
                name: "mu",
                classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
                availability: BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: "Typed-integer mean parameters are gated independently and require exact binary64 representation.",
            },
            BuiltinIntegerInputCapability {
                name: "sigma",
                classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
                availability: BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: "Typed-integer standard deviations are gated independently and require exact binary64 representation.",
            },
        ];
        pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
            [BuiltinIntegerCapabilityDescriptor {
                form: concat!($builtin, "(integer_x_or_p, integer_mu, integer_sigma)"),
                inputs: &INTEGER_INPUTS,
                computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
                output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
                overflow: BuiltinIntegerOverflowRule::Error,
                backend: BuiltinIntegerBackendRule::GatherFallback,
                overload: BuiltinIntegerOverloadKind::Multiple,
                notes: "Typed-integer inputs are RunMat-only, cross one checked binary64 distribution boundary, and return single only when another documented input is single; fallback restores through the exact owner when it can preserve the required class, otherwise automatic residency may remain host and explicit residency errors.",
            }];

        fn ensure_extensions(value: &Value, rest: &[Value]) -> BuiltinResult<()> {
            if super::is_typed_integer_value(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &INTEGER_X_EXTENSION,
                    $builtin,
                )?;
            }
            if rest.first().is_some_and(super::is_typed_integer_value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &INTEGER_MU_EXTENSION,
                    $builtin,
                )?;
            }
            if rest.get(1).is_some_and(super::is_typed_integer_value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &INTEGER_SIGMA_EXTENSION,
                    $builtin,
                )?;
            }
            if std::iter::once(value)
                .chain(rest.iter())
                .any(super::is_logical_value)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LOGICAL_INPUT_EXTENSION,
                    $builtin,
                )?;
            }
            Ok(())
        }
    };
}

async fn value_to_tensor(name: &str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| normal_error(name, format!("{name}: {err}")))?;
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| normal_error(name, format!("{name}: {err}")))?;
    tensor::integer_tensor_to_f64(tensor)
        .map_err(|err| normal_error(name, format!("{name}: {err}")))
}

fn scalar_tensor(value: f64) -> Tensor {
    Tensor::new(vec![value], vec![1, 1]).expect("scalar tensor shape is valid")
}

async fn normal_args(
    name: &str,
    first: Value,
    rest: Vec<Value>,
    allow_upper: bool,
) -> BuiltinResult<NormalArgs> {
    let mut rest = rest;
    let mut upper = false;
    if allow_upper {
        if let Some(last) = rest.last() {
            if let Some(keyword) = crate::builtins::common::random_args::keyword_of(last) {
                if keyword.eq_ignore_ascii_case("upper") {
                    upper = true;
                    rest.pop();
                }
            }
        }
    }
    if rest.len() > 2 {
        return Err(normal_error(
            name,
            format!("{name}: expected x, x, mu, or x, mu, sigma"),
        ));
    }
    let output = NormalOutputPlan::inspect(name, &first, &rest)?;
    let x = normal_value_to_tensor(name, "x or p", first).await?;
    let (mu, sigma) = match rest.as_slice() {
        [] => (scalar_tensor(0.0), scalar_tensor(1.0)),
        [mu] => (
            normal_value_to_tensor(name, "mu", mu.clone()).await?,
            scalar_tensor(1.0),
        ),
        [mu, sigma] => (
            normal_value_to_tensor(name, "mu", mu.clone()).await?,
            normal_value_to_tensor(name, "sigma", sigma.clone()).await?,
        ),
        _ => unreachable!(),
    };
    let (mut broadcasted, shape) = broadcast_tensors(name, &[&x, &mu, &sigma])?;
    let x = broadcasted.remove(0);
    let mu = broadcasted.remove(0);
    let sigma = broadcasted.remove(0);
    Ok(NormalArgs {
        x,
        mu,
        sigma,
        shape,
        upper,
        output,
    })
}

struct NormalArgs {
    x: Vec<f64>,
    mu: Vec<f64>,
    sigma: Vec<f64>,
    shape: Vec<usize>,
    upper: bool,
    output: NormalOutputPlan,
}

struct TArgs {
    x: Vec<f64>,
    nu: Vec<f64>,
    shape: Vec<usize>,
}

struct ThreeArgs {
    x: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    shape: Vec<usize>,
}

fn broadcast_pair(
    name: &str,
    lhs: &Tensor,
    rhs: &Tensor,
) -> BuiltinResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let (mut values, shape) = broadcast_tensors(name, &[lhs, rhs])?;
    let lhs = values.remove(0);
    let rhs = values.remove(0);
    Ok((lhs, rhs, shape))
}

fn broadcast_tensors(name: &str, inputs: &[&Tensor]) -> BuiltinResult<(Vec<Vec<f64>>, Vec<usize>)> {
    let Some(first) = inputs.first() else {
        return Ok((Vec::new(), vec![1, 1]));
    };
    let mut shape = first.shape.clone();
    for tensor in inputs.iter().skip(1) {
        shape = broadcast::broadcast_shapes(name, &shape, &tensor.shape)
            .map_err(|err| normal_error(name, err))?;
    }
    let mut values = Vec::with_capacity(inputs.len());
    for tensor in inputs {
        values.push(broadcast_tensor_to(name, tensor, &shape)?);
    }
    Ok((values, shape))
}

fn broadcast_tensor_to(
    name: &str,
    tensor: &Tensor,
    out_shape: &[usize],
) -> BuiltinResult<Vec<f64>> {
    let len = out_shape.iter().copied().product::<usize>();
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
            return Err(normal_error(
                name,
                format!("{name}: tensor data does not match tensor shape"),
            ));
        }
        out.push(tensor::tensor_value_f64(tensor, source_idx));
    }
    Ok(out)
}

async fn t_args(name: &str, first: Value, rest: Vec<Value>) -> BuiltinResult<TArgs> {
    if rest.len() != 1 {
        return Err(normal_error(
            name,
            format!("{name}: expected x and nu arguments"),
        ));
    }
    let x = value_to_tensor(name, first).await?;
    let nu = value_to_tensor(name, rest[0].clone()).await?;
    let (x, nu, shape) = broadcast_pair(name, &x, &nu)?;
    Ok(TArgs { x, nu, shape })
}

async fn three_args(
    name: &str,
    first: Value,
    rest: Vec<Value>,
    defaults: Option<(f64, f64)>,
) -> BuiltinResult<ThreeArgs> {
    let (a, b) = match (rest.as_slice(), defaults) {
        ([], Some((a, b))) => (scalar_tensor(a), scalar_tensor(b)),
        ([a, b], _) => (
            value_to_tensor(name, a.clone()).await?,
            value_to_tensor(name, b.clone()).await?,
        ),
        _ => {
            return Err(normal_error(
                name,
                format!("{name}: expected one or three numeric arguments"),
            ));
        }
    };
    let x = value_to_tensor(name, first).await?;
    let (mut broadcasted, shape) = broadcast_tensors(name, &[&x, &a, &b])?;
    let x = broadcasted.remove(0);
    let a = broadcasted.remove(0);
    let b = broadcasted.remove(0);
    Ok(ThreeArgs { x, a, b, shape })
}

fn finish(shape: Vec<usize>, data: Vec<f64>) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| normal_error("normal", format!("normal distribution: {err}")))
}

fn normpdf_scalar(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma.is_nan() || x.is_nan() || mu.is_nan() {
        return f64::NAN;
    }
    if sigma <= 0.0 {
        return f64::NAN;
    }
    let z = (x - mu) / sigma;
    INV_SQRT_2PI / sigma * (-0.5 * z * z).exp()
}

fn normcdf_scalar(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma.is_nan() || x.is_nan() || mu.is_nan() {
        return f64::NAN;
    }
    if sigma < 0.0 {
        return f64::NAN;
    }
    if sigma == 0.0 {
        return if x < mu { 0.0 } else { 1.0 };
    }
    0.5 * libm::erfc(-(x - mu) / (sigma * SQRT_2))
}

fn normcdf_upper_scalar(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma.is_nan() || x.is_nan() || mu.is_nan() {
        return f64::NAN;
    }
    if sigma < 0.0 {
        return f64::NAN;
    }
    if sigma == 0.0 {
        return if x < mu { 1.0 } else { 0.0 };
    }
    0.5 * libm::erfc((x - mu) / (sigma * SQRT_2))
}

fn norminv_scalar(p: f64, mu: f64, sigma: f64) -> f64 {
    if sigma.is_nan() || p.is_nan() || mu.is_nan() {
        return f64::NAN;
    }
    if sigma <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    mu - sigma * SQRT_2 * erfcinv_scalar(2.0 * p)
}

fn binocdf_scalar(x: f64, n: f64, p: f64, upper: bool) -> f64 {
    if x.is_nan()
        || n.is_nan()
        || p.is_nan()
        || n < 0.0
        || n.fract() != 0.0
        || !(0.0..=1.0).contains(&p)
    {
        return f64::NAN;
    }
    if x < 0.0 {
        return if upper { 1.0 } else { 0.0 };
    }
    let k = x.floor();
    if k >= n {
        return if upper { 0.0 } else { 1.0 };
    }
    if p == 0.0 {
        return if upper { 0.0 } else { 1.0 };
    }
    if p == 1.0 {
        return if upper { 1.0 } else { 0.0 };
    }
    if upper {
        distribution_math::regularized_beta(p, k + 1.0, n - k)
    } else {
        distribution_math::regularized_beta(1.0 - p, n - k, k + 1.0)
    }
}

fn chi2cdf_scalar(x: f64, nu: f64, upper: bool) -> f64 {
    if x.is_nan() || nu.is_nan() || nu <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return if upper { 1.0 } else { 0.0 };
    }
    if upper {
        distribution_math::regularized_gamma_q(nu / 2.0, x / 2.0)
    } else {
        distribution_math::regularized_gamma_p(nu / 2.0, x / 2.0)
    }
}

fn wblinv_scalar(p: f64, a: f64, b: f64) -> f64 {
    if p.is_nan() || a.is_nan() || b.is_nan() || a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    a * (-(-p).ln_1p()).powf(1.0 / b)
}

fn chi2inv_scalar(p: f64, nu: f64) -> f64 {
    if p.is_nan() || nu.is_nan() || nu <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let mut lo = 0.0;
    let mut hi = nu.max(1.0);
    let mut iterations = 0;
    while chi2cdf_scalar(hi, nu, false) < p {
        hi *= 2.0;
        iterations += 1;
        if !hi.is_finite() || iterations > 2048 {
            return f64::INFINITY;
        }
    }
    for _ in 0..160 {
        let mid = 0.5 * (lo + hi);
        if chi2cdf_scalar(mid, nu, false) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

fn binoinv_scalar(probability: f64, n: f64, success_probability: f64) -> f64 {
    if probability.is_nan()
        || n.is_nan()
        || success_probability.is_nan()
        || n < 0.0
        || n.fract() != 0.0
        || !(0.0..=1.0).contains(&probability)
        || !(0.0..=1.0).contains(&success_probability)
    {
        return f64::NAN;
    }
    if probability == 0.0 || success_probability == 0.0 {
        return 0.0;
    }
    if probability == 1.0 || success_probability == 1.0 {
        return n;
    }
    let mut lo = 0.0;
    let mut hi = n;
    while lo < hi {
        let mid = ((lo + hi) / 2.0).floor();
        if binocdf_scalar(mid, n, success_probability, false) >= probability {
            hi = mid;
        } else {
            lo = mid + 1.0;
        }
    }
    lo
}

fn expinv_scalar(p: f64, mu: f64) -> f64 {
    if p.is_nan() || mu.is_nan() || mu <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    -mu * (-p).ln_1p()
}

fn unifinv_scalar(p: f64, a: f64, b: f64) -> f64 {
    if p.is_nan() || a.is_nan() || b.is_nan() || a > b || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    a + p * (b - a)
}

fn logninv_scalar(p: f64, mu: f64, sigma: f64) -> f64 {
    let normal = norminv_scalar(p, mu, sigma);
    if normal.is_nan() {
        f64::NAN
    } else {
        normal.exp()
    }
}

fn gaminv_scalar(p: f64, shape: f64, scale: f64) -> f64 {
    if p.is_nan()
        || shape.is_nan()
        || scale.is_nan()
        || shape <= 0.0
        || scale <= 0.0
        || !(0.0..=1.0).contains(&p)
    {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    invert_continuous_positive(p, shape * scale, |x| {
        distribution_math::regularized_gamma_p(shape, x / scale)
    })
}

fn betainv_scalar(p: f64, a: f64, b: f64) -> f64 {
    if p.is_nan() || a.is_nan() || b.is_nan() || a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }
    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..180 {
        let mid = 0.5 * (lo + hi);
        if distribution_math::regularized_beta(mid, a, b) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

fn finv_scalar(p: f64, v1: f64, v2: f64) -> f64 {
    if p.is_nan()
        || v1.is_nan()
        || v2.is_nan()
        || v1 <= 0.0
        || v2 <= 0.0
        || !(0.0..=1.0).contains(&p)
    {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    invert_continuous_positive(p, 1.0, |x| {
        let scaled = v1 * x;
        let beta_x = scaled / (scaled + v2);
        distribution_math::regularized_beta(beta_x, v1 / 2.0, v2 / 2.0)
    })
}

fn poissinv_scalar(p: f64, lambda: f64) -> f64 {
    if p.is_nan() || lambda.is_nan() || lambda < 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p == 0.0 || lambda == 0.0 {
        return 0.0;
    }
    let mut lo = 0.0;
    let mut hi = lambda.ceil().max(1.0);
    let mut iterations = 0;
    while poisscdf_scalar(hi, lambda) < p {
        hi *= 2.0;
        iterations += 1;
        if !hi.is_finite() || iterations > 2048 {
            return f64::INFINITY;
        }
    }
    while lo < hi {
        let mid = ((lo + hi) / 2.0).floor();
        if poisscdf_scalar(mid, lambda) >= p {
            hi = mid;
        } else {
            lo = mid + 1.0;
        }
    }
    lo
}

fn poisscdf_scalar(k: f64, lambda: f64) -> f64 {
    if k.is_nan() || lambda.is_nan() || lambda < 0.0 {
        return f64::NAN;
    }
    if k < 0.0 {
        return 0.0;
    }
    if lambda == 0.0 {
        return 1.0;
    }
    distribution_math::regularized_gamma_q(k.floor() + 1.0, lambda)
}

fn invert_continuous_positive<F>(p: f64, initial_hi: f64, mut cdf: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    let mut lo = 0.0;
    let mut hi = initial_hi.max(1.0);
    let mut iterations = 0;
    while cdf(hi) < p {
        hi *= 2.0;
        iterations += 1;
        if !hi.is_finite() || iterations > 2048 {
            return f64::INFINITY;
        }
    }
    for _ in 0..180 {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

pub mod normpdf;

pub mod normcdf;

pub mod norminv;

pub mod tpdf;

pub mod tcdf;

pub mod tinv;

pub mod binocdf;

pub mod chi2cdf;

pub mod wblinv;

pub mod icdf;

#[cfg(test)]
mod tests;
