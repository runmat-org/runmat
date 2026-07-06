//! Probability distribution compatibility helpers.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{broadcast, tensor};
use crate::builtins::math::elementwise::erfcinv::erfcinv_scalar;
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
        inputs: &INPUTS_X_NU,
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

async fn value_to_tensor(name: &str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| normal_error(name, format!("{name}: {err}")))?;
    tensor::value_into_tensor_for(name, gathered)
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
    let x = value_to_tensor(name, first).await?;
    let (mu, sigma) = match rest.as_slice() {
        [] => (scalar_tensor(0.0), scalar_tensor(1.0)),
        [mu] => (value_to_tensor(name, mu.clone()).await?, scalar_tensor(1.0)),
        [mu, sigma] => (
            value_to_tensor(name, mu.clone()).await?,
            value_to_tensor(name, sigma.clone()).await?,
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
    })
}

struct NormalArgs {
    x: Vec<f64>,
    mu: Vec<f64>,
    sigma: Vec<f64>,
    shape: Vec<usize>,
    upper: bool,
}

struct TArgs {
    x: Vec<f64>,
    nu: Vec<f64>,
    shape: Vec<usize>,
    upper: bool,
}

struct ThreeArgs {
    x: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    shape: Vec<usize>,
    upper: bool,
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
    let in_shape = align_shape(&tensor.shape, out_shape.len());
    let strides = broadcast::compute_strides(&in_shape);
    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        let source_idx = broadcast::broadcast_index(idx, out_shape, &in_shape, &strides);
        let Some(value) = tensor.data.get(source_idx) else {
            return Err(normal_error(
                name,
                format!("{name}: tensor data does not match tensor shape"),
            ));
        };
        out.push(*value);
    }
    Ok(out)
}

fn align_shape(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut aligned = Vec::with_capacity(rank);
    aligned.extend(std::iter::repeat_n(1, rank.saturating_sub(shape.len())));
    aligned.extend_from_slice(shape);
    aligned
}

async fn t_args(
    name: &str,
    first: Value,
    rest: Vec<Value>,
    allow_upper: bool,
) -> BuiltinResult<TArgs> {
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
    if rest.len() != 1 {
        return Err(normal_error(
            name,
            format!("{name}: expected x and nu arguments"),
        ));
    }
    let x = value_to_tensor(name, first).await?;
    let nu = value_to_tensor(name, rest[0].clone()).await?;
    let (x, nu, shape) = broadcast_pair(name, &x, &nu)?;
    Ok(TArgs {
        x,
        nu,
        shape,
        upper,
    })
}

async fn three_args(
    name: &str,
    first: Value,
    rest: Vec<Value>,
    defaults: Option<(f64, f64)>,
    allow_upper: bool,
) -> BuiltinResult<ThreeArgs> {
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
    Ok(ThreeArgs {
        x,
        a,
        b,
        shape,
        upper,
    })
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

pub mod normpdf {
    use super::*;
    normal_descriptor!("normpdf", NORMAL_SIGNATURES_X);

    #[runtime_builtin(
        name = "normpdf",
        category = "stats/summary",
        summary = "Evaluate the normal probability density function.",
        keywords = "normpdf,normal,gaussian,pdf,statistics",
        type_resolver(super::normal_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::distributions::normpdf"
    )]
    pub(crate) async fn normpdf_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::normal_args("normpdf", value, rest, false).await?;
        let data = args
            .x
            .iter()
            .zip(args.mu.iter())
            .zip(args.sigma.iter())
            .map(|((x, mu), sigma)| super::normpdf_scalar(*x, *mu, *sigma))
            .collect();
        super::finish(args.shape, data)
    }
}

pub mod normcdf {
    use super::*;
    normal_descriptor!("normcdf", NORMAL_CDF_SIGNATURES);

    #[runtime_builtin(
        name = "normcdf",
        category = "stats/summary",
        summary = "Evaluate the normal cumulative distribution function.",
        keywords = "normcdf,normal,gaussian,cdf,statistics",
        type_resolver(super::normal_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::distributions::normcdf"
    )]
    pub(crate) async fn normcdf_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::normal_args("normcdf", value, rest, true).await?;
        let data = args
            .x
            .iter()
            .zip(args.mu.iter())
            .zip(args.sigma.iter())
            .map(|((x, mu), sigma)| {
                if args.upper {
                    super::normcdf_upper_scalar(*x, *mu, *sigma)
                } else {
                    super::normcdf_scalar(*x, *mu, *sigma)
                }
            })
            .collect();
        super::finish(args.shape, data)
    }
}

pub mod norminv {
    use super::*;
    normal_descriptor!("norminv", NORMAL_INV_SIGNATURES);

    #[runtime_builtin(
        name = "norminv",
        category = "stats/summary",
        summary = "Evaluate the inverse normal cumulative distribution function.",
        keywords = "norminv,normal,gaussian,inverse,cdf,statistics",
        type_resolver(super::normal_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::distributions::norminv"
    )]
    pub(crate) async fn norminv_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::normal_args("norminv", value, rest, false).await?;
        let data = args
            .x
            .iter()
            .zip(args.mu.iter())
            .zip(args.sigma.iter())
            .map(|((p, mu), sigma)| super::norminv_scalar(*p, *mu, *sigma))
            .collect();
        super::finish(args.shape, data)
    }
}

pub mod tpdf {
    use super::*;
    normal_descriptor!("tpdf", T_PDF_SIGNATURES);

    #[runtime_builtin(
        name = "tpdf",
        category = "stats/summary",
        summary = "Evaluate the Student's t probability density function.",
        keywords = "tpdf,student t,pdf,statistics,distribution",
        type_resolver(super::normal_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::distributions::tpdf"
    )]
    pub(crate) async fn tpdf_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::t_args("tpdf", value, rest, false).await?;
        let data = args
            .x
            .iter()
            .zip(args.nu.iter())
            .map(|(x, nu)| distribution_math::student_t_pdf(*x, *nu))
            .collect();
        super::finish(args.shape, data)
    }
}

pub mod tcdf {
    use super::*;
    normal_descriptor!("tcdf", T_CDF_SIGNATURES);

    #[runtime_builtin(
        name = "tcdf",
        category = "stats/summary",
        summary = "Evaluate the Student's t cumulative distribution function.",
        keywords = "tcdf,student t,cdf,upper tail,statistics,distribution",
        type_resolver(super::normal_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::distributions::tcdf"
    )]
    pub(crate) async fn tcdf_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::t_args("tcdf", value, rest, true).await?;
        let data = args
            .x
            .iter()
            .zip(args.nu.iter())
            .map(|(x, nu)| {
                if args.upper {
                    distribution_math::student_t_cdf_upper(*x, *nu)
                } else {
                    distribution_math::student_t_cdf(*x, *nu)
                }
            })
            .collect();
        super::finish(args.shape, data)
    }
}

pub mod tinv {
    use super::*;
    normal_descriptor!("tinv", T_INV_SIGNATURES);

    #[runtime_builtin(
        name = "tinv",
        category = "stats/summary",
        summary = "Evaluate the inverse Student's t cumulative distribution function.",
        keywords = "tinv,student t,inverse,cdf,statistics,distribution",
        type_resolver(super::normal_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::distributions::tinv"
    )]
    pub(crate) async fn tinv_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::t_args("tinv", value, rest, false).await?;
        let data = args
            .x
            .iter()
            .zip(args.nu.iter())
            .map(|(p, nu)| distribution_math::student_t_inv(*p, *nu))
            .collect();
        super::finish(args.shape, data)
    }
}

pub mod binocdf {
    use super::*;
    normal_descriptor!("binocdf", BINOCDF_SIGNATURES);

    #[runtime_builtin(
        name = "binocdf",
        category = "stats/summary",
        summary = "Evaluate the binomial cumulative distribution function.",
        keywords = "binocdf,binomial,cdf,upper tail,statistics,distribution",
        type_resolver(super::normal_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::distributions::binocdf"
    )]
    pub(crate) async fn binocdf_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::three_args("binocdf", value, rest, None, true).await?;
        let data = args
            .x
            .iter()
            .zip(args.a.iter())
            .zip(args.b.iter())
            .map(|((x, n), p)| super::binocdf_scalar(*x, *n, *p, args.upper))
            .collect();
        super::finish(args.shape, data)
    }
}

pub mod chi2cdf {
    use super::*;
    normal_descriptor!("chi2cdf", CHI2CDF_SIGNATURES);

    #[runtime_builtin(
        name = "chi2cdf",
        category = "stats/summary",
        summary = "Evaluate the chi-square cumulative distribution function.",
        keywords = "chi2cdf,chi-square,cdf,upper tail,statistics,distribution",
        type_resolver(super::normal_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::distributions::chi2cdf"
    )]
    pub(crate) async fn chi2cdf_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::t_args("chi2cdf", value, rest, true).await?;
        let data = args
            .x
            .iter()
            .zip(args.nu.iter())
            .map(|(x, nu)| super::chi2cdf_scalar(*x, *nu, args.upper))
            .collect();
        super::finish(args.shape, data)
    }
}

pub mod wblinv {
    use super::*;
    normal_descriptor!("wblinv", WBLINV_SIGNATURES);

    #[runtime_builtin(
        name = "wblinv",
        category = "stats/summary",
        summary = "Evaluate the inverse Weibull cumulative distribution function.",
        keywords = "wblinv,weibull,inverse,cdf,statistics,distribution",
        type_resolver(super::normal_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::distributions::wblinv"
    )]
    pub(crate) async fn wblinv_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::three_args("wblinv", value, rest, Some((1.0, 1.0)), false).await?;
        let data = args
            .x
            .iter()
            .zip(args.a.iter())
            .zip(args.b.iter())
            .map(|((p, a), b)| super::wblinv_scalar(*p, *a, *b))
            .collect();
        super::finish(args.shape, data)
    }
}

pub mod icdf {
    use super::*;

    const ERRORS: [BuiltinErrorDescriptor; 2] = [
        BuiltinErrorDescriptor {
            code: "RM.icdf.INVALID_ARGUMENT",
            identifier: Some("RunMat:icdf:InvalidArgument"),
            when: ERROR_INVALID_ARGUMENT.when,
            message: "icdf: invalid argument",
        },
        BuiltinErrorDescriptor {
            code: "RM.icdf.INTERNAL",
            identifier: Some("RunMat:icdf:Internal"),
            when: ERROR_INTERNAL.when,
            message: "icdf: internal error",
        },
    ];

    pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
        signatures: &ICDF_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

    #[derive(Clone, Copy)]
    enum IcdfDistribution {
        Normal,
        StudentT,
        Weibull,
        ChiSquare,
        Binomial,
        Gamma,
        Exponential,
        Poisson,
        Uniform,
        Lognormal,
        Beta,
        F,
    }

    #[runtime_builtin(
        name = "icdf",
        category = "stats/summary",
        summary = "Evaluate inverse cumulative distribution functions by distribution name.",
        keywords = "icdf,inverse,cdf,normal,student t,weibull,binomial,chi-square,gamma,exponential,poisson,uniform,lognormal,beta,f,statistics",
        type_resolver(super::icdf_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::distributions::icdf"
    )]
    pub(crate) async fn icdf_builtin(
        name: Value,
        p: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<Value> {
        if matches!(name, Value::Object(_)) {
            if !rest.is_empty() {
                return Err(super::normal_error(
                    "icdf",
                    "icdf: fitted distribution object form accepts exactly two inputs",
                ));
            }
            return crate::builtins::stats::summary::fitdist::icdf_probability_distribution(
                name, p,
            )
            .await;
        }

        let distribution = parse_distribution_name(&name)?;
        match distribution {
            IcdfDistribution::Normal => normal_icdf(p, rest).await,
            IcdfDistribution::StudentT => student_t_icdf(p, rest).await,
            IcdfDistribution::Weibull => weibull_icdf(p, rest).await,
            IcdfDistribution::ChiSquare => chi_square_icdf(p, rest).await,
            IcdfDistribution::Binomial => binomial_icdf(p, rest).await,
            IcdfDistribution::Gamma => gamma_icdf(p, rest).await,
            IcdfDistribution::Exponential => exponential_icdf(p, rest).await,
            IcdfDistribution::Poisson => poisson_icdf(p, rest).await,
            IcdfDistribution::Uniform => uniform_icdf(p, rest).await,
            IcdfDistribution::Lognormal => lognormal_icdf(p, rest).await,
            IcdfDistribution::Beta => beta_icdf(p, rest).await,
            IcdfDistribution::F => f_icdf(p, rest).await,
        }
    }

    fn parse_distribution_name(value: &Value) -> BuiltinResult<IcdfDistribution> {
        let Some(keyword) = crate::builtins::common::random_args::keyword_of(value) else {
            return Err(super::normal_error(
                "icdf",
                "icdf: distribution name must be a string scalar",
            ));
        };
        let normalized = keyword
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>();
        match normalized.as_str() {
            "normal" | "norm" | "gaussian" => Ok(IcdfDistribution::Normal),
            "t" | "tdistribution" | "studentt" | "student" => Ok(IcdfDistribution::StudentT),
            "weibull" | "wbl" => Ok(IcdfDistribution::Weibull),
            "chisquare" | "chi2" | "chisquared" => Ok(IcdfDistribution::ChiSquare),
            "binomial" | "bino" => Ok(IcdfDistribution::Binomial),
            "gamma" | "gam" => Ok(IcdfDistribution::Gamma),
            "exponential" | "exp" => Ok(IcdfDistribution::Exponential),
            "poisson" | "poiss" => Ok(IcdfDistribution::Poisson),
            "uniform" | "unif" => Ok(IcdfDistribution::Uniform),
            "lognormal" | "logn" => Ok(IcdfDistribution::Lognormal),
            "beta" => Ok(IcdfDistribution::Beta),
            "f" | "fdist" | "fdistribution" => Ok(IcdfDistribution::F),
            _ => Err(super::normal_error(
                "icdf",
                format!("icdf: unsupported distribution '{keyword}'"),
            )),
        }
    }

    async fn normal_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::normal_args("icdf", p, rest, false).await?;
        let data = args
            .x
            .iter()
            .zip(args.mu.iter())
            .zip(args.sigma.iter())
            .map(|((p, mu), sigma)| super::norminv_scalar(*p, *mu, *sigma))
            .collect();
        super::finish(args.shape, data)
    }

    async fn student_t_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::t_args("icdf", p, rest, false).await?;
        let data = args
            .x
            .iter()
            .zip(args.nu.iter())
            .map(|(p, nu)| distribution_math::student_t_inv(*p, *nu))
            .collect();
        super::finish(args.shape, data)
    }

    async fn weibull_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::three_args("icdf", p, rest, Some((1.0, 1.0)), false).await?;
        let data = args
            .x
            .iter()
            .zip(args.a.iter())
            .zip(args.b.iter())
            .map(|((p, a), b)| super::wblinv_scalar(*p, *a, *b))
            .collect();
        super::finish(args.shape, data)
    }

    async fn chi_square_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::t_args("icdf", p, rest, false).await?;
        let data = args
            .x
            .iter()
            .zip(args.nu.iter())
            .map(|(p, nu)| super::chi2inv_scalar(*p, *nu))
            .collect();
        super::finish(args.shape, data)
    }

    async fn binomial_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::three_args("icdf", p, rest, None, false).await?;
        let data = args
            .x
            .iter()
            .zip(args.a.iter())
            .zip(args.b.iter())
            .map(|((probability, n), success_probability)| {
                super::binoinv_scalar(*probability, *n, *success_probability)
            })
            .collect();
        super::finish(args.shape, data)
    }

    async fn gamma_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::three_args("icdf", p, rest, None, false).await?;
        let data = args
            .x
            .iter()
            .zip(args.a.iter())
            .zip(args.b.iter())
            .map(|((p, shape), scale)| super::gaminv_scalar(*p, *shape, *scale))
            .collect();
        super::finish(args.shape, data)
    }

    async fn exponential_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = one_param_args("Exponential", p, rest).await?;
        let data = args
            .x
            .iter()
            .zip(args.param.iter())
            .map(|(p, mu)| super::expinv_scalar(*p, *mu))
            .collect();
        super::finish(args.shape, data)
    }

    async fn poisson_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = one_param_args("Poisson", p, rest).await?;
        let data = args
            .x
            .iter()
            .zip(args.param.iter())
            .map(|(p, lambda)| super::poissinv_scalar(*p, *lambda))
            .collect();
        super::finish(args.shape, data)
    }

    async fn uniform_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::three_args("icdf", p, rest, Some((0.0, 1.0)), false).await?;
        let data = args
            .x
            .iter()
            .zip(args.a.iter())
            .zip(args.b.iter())
            .map(|((p, a), b)| super::unifinv_scalar(*p, *a, *b))
            .collect();
        super::finish(args.shape, data)
    }

    async fn lognormal_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::normal_args("icdf", p, rest, false).await?;
        let data = args
            .x
            .iter()
            .zip(args.mu.iter())
            .zip(args.sigma.iter())
            .map(|((p, mu), sigma)| super::logninv_scalar(*p, *mu, *sigma))
            .collect();
        super::finish(args.shape, data)
    }

    async fn beta_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::three_args("icdf", p, rest, None, false).await?;
        let data = args
            .x
            .iter()
            .zip(args.a.iter())
            .zip(args.b.iter())
            .map(|((p, a), b)| super::betainv_scalar(*p, *a, *b))
            .collect();
        super::finish(args.shape, data)
    }

    async fn f_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::three_args("icdf", p, rest, None, false).await?;
        let data = args
            .x
            .iter()
            .zip(args.a.iter())
            .zip(args.b.iter())
            .map(|((p, v1), v2)| super::finv_scalar(*p, *v1, *v2))
            .collect();
        super::finish(args.shape, data)
    }

    struct OneParamArgs {
        x: Vec<f64>,
        param: Vec<f64>,
        shape: Vec<usize>,
    }

    async fn one_param_args(
        distribution: &str,
        p: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<OneParamArgs> {
        if rest.len() != 1 {
            return Err(super::normal_error(
                "icdf",
                format!("icdf: {distribution} distribution expects one parameter"),
            ));
        }
        let x = super::value_to_tensor("icdf", p).await?;
        let param = super::value_to_tensor("icdf", rest[0].clone()).await?;
        let (x, param, shape) = super::broadcast_pair("icdf", &x, &param)?;
        Ok(OneParamArgs { x, param, shape })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "actual={actual} expected={expected}"
        );
    }

    fn icdf_scalar(name: &str, p: f64, rest: Vec<Value>) -> f64 {
        match block_on(icdf::icdf_builtin(Value::from(name), Value::Num(p), rest)).unwrap() {
            Value::Num(value) => value,
            other => panic!("expected scalar icdf result, got {other:?}"),
        }
    }

    #[test]
    fn normal_distribution_scalar_values() {
        let pdf = block_on(normpdf::normpdf_builtin(Value::Num(0.0), Vec::new())).unwrap();
        match pdf {
            Value::Num(value) => assert_close(value, INV_SQRT_2PI, 1e-12),
            other => panic!("expected scalar pdf, got {other:?}"),
        }
        let cdf = block_on(normcdf::normcdf_builtin(Value::Num(0.0), Vec::new())).unwrap();
        match cdf {
            Value::Num(value) => assert_close(value, 0.5, 1e-12),
            other => panic!("expected scalar cdf, got {other:?}"),
        }
        let inv = block_on(norminv::norminv_builtin(Value::Num(0.5), Vec::new())).unwrap();
        match inv {
            Value::Num(value) => assert_close(value, 0.0, 1e-10),
            other => panic!("expected scalar inv, got {other:?}"),
        }
    }

    #[test]
    fn normal_distribution_broadcasts_parameters() {
        let x = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
        let out = block_on(normcdf::normcdf_builtin(
            x,
            vec![Value::Num(0.0), Value::Num(1.0)],
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_close(tensor.data[0], 0.5, 1e-12);
                assert_close(tensor.data[1], 0.841_344_746_068_543, 1e-12);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn normal_distribution_accepts_mu_only_and_upper_tail() {
        let cdf = block_on(normcdf::normcdf_builtin(
            Value::Num(2.0),
            vec![Value::Num(1.0)],
        ))
        .unwrap();
        match cdf {
            Value::Num(value) => assert_close(value, 0.841_344_746_068_543, 1e-12),
            other => panic!("expected scalar cdf, got {other:?}"),
        }

        let upper = block_on(normcdf::normcdf_builtin(
            Value::Num(8.0),
            vec![Value::Num(0.0), Value::Num(1.0), Value::from("upper")],
        ))
        .unwrap();
        match upper {
            Value::Num(value) => assert_close(value, 6.220_960_574_271_784e-16, 1e-28),
            other => panic!("expected scalar upper cdf, got {other:?}"),
        }

        let pdf = block_on(normpdf::normpdf_builtin(
            Value::Num(2.0),
            vec![Value::Num(1.0)],
        ))
        .unwrap();
        match pdf {
            Value::Num(value) => assert_close(value, INV_SQRT_2PI * (-0.5f64).exp(), 1e-12),
            other => panic!("expected scalar pdf, got {other:?}"),
        }

        let inv = block_on(norminv::norminv_builtin(
            Value::Num(0.5),
            vec![Value::Num(2.0)],
        ))
        .unwrap();
        match inv {
            Value::Num(value) => assert_close(value, 2.0, 1e-10),
            other => panic!("expected scalar inv, got {other:?}"),
        }
    }

    #[test]
    fn icdf_dispatches_supported_distribution_names() {
        let normal = block_on(icdf::icdf_builtin(
            Value::from("Normal"),
            Value::Num(0.5),
            vec![Value::Num(10.0), Value::Num(2.0)],
        ))
        .unwrap();
        match normal {
            Value::Num(value) => assert_close(value, 10.0, 1e-10),
            other => panic!("expected scalar normal icdf, got {other:?}"),
        }

        let student_t = block_on(icdf::icdf_builtin(
            Value::from("Student t"),
            Value::Num(0.95),
            vec![Value::Num(50.0)],
        ))
        .unwrap();
        match student_t {
            Value::Num(value) => assert_close(value, 1.675_905, 1e-6),
            other => panic!("expected scalar t icdf, got {other:?}"),
        }

        let weibull = block_on(icdf::icdf_builtin(
            Value::from("Weibull"),
            Value::Num(0.5),
            vec![Value::Num(3.0), Value::Num(4.0)],
        ))
        .unwrap();
        match weibull {
            Value::Num(value) => {
                assert_close(value, 3.0 * std::f64::consts::LN_2.powf(0.25), 1e-12)
            }
            other => panic!("expected scalar weibull icdf, got {other:?}"),
        }
    }

    #[test]
    fn icdf_broadcasts_and_inverts_chisquare_and_binomial() {
        let probabilities = Value::Tensor(Tensor::new(vec![0.5, 0.95], vec![1, 2]).unwrap());
        let normal = block_on(icdf::icdf_builtin(
            Value::from("norm"),
            probabilities,
            vec![Value::Num(1.0), Value::Num(2.0)],
        ))
        .unwrap();
        match normal {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_close(tensor.data[0], 1.0, 1e-10);
                assert_close(tensor.data[1], 4.289_707_253_902_944, 1e-10);
            }
            other => panic!("expected tensor normal icdf, got {other:?}"),
        }

        let chi_square = block_on(icdf::icdf_builtin(
            Value::from("Chi-square"),
            Value::Num(0.5),
            vec![Value::Num(2.0)],
        ))
        .unwrap();
        match chi_square {
            Value::Num(value) => assert_close(value, 2.0 * std::f64::consts::LN_2, 1e-12),
            other => panic!("expected scalar chi-square icdf, got {other:?}"),
        }

        let binomial = block_on(icdf::icdf_builtin(
            Value::from("Binomial"),
            Value::Num(0.75),
            vec![Value::Num(10.0), Value::Num(0.5)],
        ))
        .unwrap();
        match binomial {
            Value::Num(value) => assert_close(value, 6.0, 0.0),
            other => panic!("expected scalar binomial icdf, got {other:?}"),
        }
    }

    #[test]
    fn icdf_dispatches_common_distribution_families() {
        assert_close(
            icdf_scalar("Exponential", 0.5, vec![Value::Num(4.0)]),
            4.0 * std::f64::consts::LN_2,
            1e-12,
        );
        assert_close(
            icdf_scalar("Uniform", 0.25, vec![Value::Num(10.0), Value::Num(20.0)]),
            12.5,
            1e-12,
        );
        assert_close(
            icdf_scalar("Lognormal", 0.5, vec![Value::Num(1.0), Value::Num(2.0)]),
            std::f64::consts::E,
            1e-10,
        );
        assert_close(icdf_scalar("Poisson", 0.8, vec![Value::Num(3.0)]), 4.0, 0.0);
        assert_close(
            icdf_scalar("Beta", 0.5, vec![Value::Num(2.0), Value::Num(2.0)]),
            0.5,
            1e-12,
        );

        let gamma = icdf_scalar("Gamma", 0.6, vec![Value::Num(2.0), Value::Num(3.0)]);
        assert_close(
            distribution_math::regularized_gamma_p(2.0, gamma / 3.0),
            0.6,
            1e-10,
        );

        let f = icdf_scalar("F", 0.6, vec![Value::Num(5.0), Value::Num(10.0)]);
        let beta_x = 5.0 * f / (5.0 * f + 10.0);
        assert_close(
            distribution_math::regularized_beta(beta_x, 2.5, 5.0),
            0.6,
            1e-10,
        );
    }

    #[test]
    fn icdf_handles_boundaries_invalid_params_and_bad_broadcasts() {
        assert_close(
            icdf_scalar("Uniform", 0.0, vec![Value::Num(2.0), Value::Num(5.0)]),
            2.0,
            0.0,
        );
        assert_close(
            icdf_scalar("Uniform", 1.0, vec![Value::Num(2.0), Value::Num(5.0)]),
            5.0,
            0.0,
        );
        assert_close(
            icdf_scalar("Gamma", 0.0, vec![Value::Num(2.0), Value::Num(3.0)]),
            0.0,
            0.0,
        );
        assert!(icdf_scalar("Gamma", 1.0, vec![Value::Num(2.0), Value::Num(3.0)]).is_infinite());
        assert_close(
            icdf_scalar("Binomial", 0.0, vec![Value::Num(10.0), Value::Num(0.5)]),
            0.0,
            0.0,
        );
        assert_close(
            icdf_scalar("Binomial", 1.0, vec![Value::Num(10.0), Value::Num(0.5)]),
            10.0,
            0.0,
        );
        assert!(icdf_scalar("Normal", -0.1, Vec::new()).is_nan());
        assert!(icdf_scalar("Gamma", 0.5, vec![Value::Num(0.0), Value::Num(3.0)]).is_nan());
        assert!(icdf_scalar("Uniform", 0.5, vec![Value::Num(5.0), Value::Num(2.0)]).is_nan());

        let err = block_on(icdf::icdf_builtin(
            Value::from("Exponential"),
            Value::Tensor(Tensor::new(vec![0.1, 0.2], vec![1, 2]).unwrap()),
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap(),
            )],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:icdf:InvalidArgument"));
    }

    #[test]
    fn icdf_broadcasts_later_parameter_shapes() {
        let normal = block_on(icdf::icdf_builtin(
            Value::from("Normal"),
            Value::Num(0.5),
            vec![
                Value::Num(0.0),
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            ],
        ))
        .unwrap();
        match normal {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_close(tensor.data[0], 0.0, 1e-10);
                assert_close(tensor.data[1], 0.0, 1e-10);
            }
            other => panic!("expected tensor normal icdf, got {other:?}"),
        }

        let uniform = block_on(icdf::icdf_builtin(
            Value::from("Uniform"),
            Value::Num(0.5),
            vec![
                Value::Num(0.0),
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            ],
        ))
        .unwrap();
        match uniform {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_close(tensor.data[0], 0.5, 1e-12);
                assert_close(tensor.data[1], 1.0, 1e-12);
            }
            other => panic!("expected tensor uniform icdf, got {other:?}"),
        }
    }

    #[test]
    fn icdf_type_uses_broadcasted_numeric_argument_shape() {
        let probability_type = Type::Tensor {
            shape: Some(vec![Some(1), Some(2)]),
        };
        let inferred = icdf_type(
            &[Type::String, probability_type.clone(), Type::Num, Type::Num],
            &ResolveContext::default(),
        );
        assert_eq!(inferred, probability_type);
        assert_eq!(
            icdf_type(&[Type::String, Type::Num], &ResolveContext::default()),
            Type::Num
        );
        assert_eq!(
            icdf_type(
                &[
                    Type::String,
                    Type::Num,
                    Type::Tensor {
                        shape: Some(vec![Some(1), Some(2)])
                    }
                ],
                &ResolveContext::default()
            ),
            Type::Tensor {
                shape: Some(vec![Some(1), Some(2)])
            }
        );
    }

    #[test]
    fn icdf_poisson_large_lambda_is_bounded_and_finite() {
        let value = icdf_scalar("Poisson", 0.5, vec![Value::Num(1000.0)]);
        assert!(value.is_finite());
        assert!(value >= 0.0);
        assert!(poisscdf_scalar(value, 1000.0) >= 0.5);
        assert!(value == 0.0 || poisscdf_scalar(value - 1.0, 1000.0) < 0.5);
    }

    #[test]
    fn icdf_rejects_unknown_distribution_name() {
        let err = block_on(icdf::icdf_builtin(
            Value::from("not_a_distribution"),
            Value::Num(0.5),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(err.message().contains("unsupported distribution"));
        assert_eq!(err.identifier(), Some("RunMat:icdf:InvalidArgument"));
    }

    #[test]
    fn normal_distribution_rejects_invalid_sigma_with_nan_outputs() {
        let pdf = block_on(normpdf::normpdf_builtin(
            Value::Num(0.0),
            vec![Value::Num(0.0), Value::Num(0.0)],
        ))
        .unwrap();
        assert!(matches!(pdf, Value::Num(value) if value.is_nan()));

        let inv = block_on(norminv::norminv_builtin(
            Value::Num(0.5),
            vec![Value::Num(0.0), Value::Num(0.0)],
        ))
        .unwrap();
        assert!(matches!(inv, Value::Num(value) if value.is_nan()));

        let cdf = block_on(normcdf::normcdf_builtin(
            Value::Num(0.0),
            vec![Value::Num(1.0), Value::Num(0.0)],
        ))
        .unwrap();
        assert_eq!(cdf, Value::Num(0.0));
    }

    #[test]
    fn student_t_distribution_scalar_values() {
        let pdf = block_on(tpdf::tpdf_builtin(Value::Num(0.0), vec![Value::Num(1.0)])).unwrap();
        match pdf {
            Value::Num(value) => assert_close(value, std::f64::consts::FRAC_1_PI, 1e-12),
            other => panic!("expected scalar pdf, got {other:?}"),
        }

        let cdf = block_on(tcdf::tcdf_builtin(Value::Num(0.0), vec![Value::Num(10.0)])).unwrap();
        match cdf {
            Value::Num(value) => assert_close(value, 0.5, 1e-12),
            other => panic!("expected scalar cdf, got {other:?}"),
        }

        let inv = block_on(tinv::tinv_builtin(Value::Num(0.95), vec![Value::Num(50.0)])).unwrap();
        match inv {
            Value::Num(value) => assert_close(value, 1.675_905, 1e-6),
            other => panic!("expected scalar inv, got {other:?}"),
        }
    }

    #[test]
    fn student_t_distribution_broadcasts_and_upper_tail() {
        let x = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap());
        let nu = Value::Tensor(Tensor::new(vec![1.0, 5.0, f64::INFINITY], vec![1, 3]).unwrap());
        let out = block_on(tcdf::tcdf_builtin(x, vec![nu])).unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_close(tensor.data[0], 0.5, 1e-12);
                assert_close(tensor.data[1], 0.818_391_266, 1e-9);
                assert_close(tensor.data[2], 0.977_249_868, 1e-9);
            }
            other => panic!("expected tensor cdf, got {other:?}"),
        }

        let upper = block_on(tcdf::tcdf_builtin(
            Value::Num(10.0),
            vec![Value::Num(99.0), Value::from("upper")],
        ))
        .unwrap();
        match upper {
            Value::Num(value) => assert_close(value, 5.469_9e-17, 1e-20),
            other => panic!("expected scalar upper cdf, got {other:?}"),
        }
    }

    #[test]
    fn student_t_distribution_rejects_bad_shapes_and_returns_nan_for_bad_parameters() {
        let err = block_on(tpdf::tpdf_builtin(
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap(),
            )],
        ))
        .unwrap_err();
        assert!(err.message().contains("tpdf"));
        assert_eq!(err.identifier(), Some("RunMat:tpdf:InvalidArgument"));

        let out = block_on(tcdf::tcdf_builtin(Value::Num(0.0), vec![Value::Num(-1.0)])).unwrap();
        assert!(matches!(out, Value::Num(value) if value.is_nan()));
    }

    #[test]
    fn student_t_distribution_extreme_tails_remain_representable() {
        let lower = block_on(tinv::tinv_builtin(
            Value::Num(1.0e-20),
            vec![Value::Num(1.0)],
        ))
        .unwrap();
        match lower {
            Value::Num(value) => {
                assert!(value.is_finite());
                assert_close(value / 1.0e19, -3.183_098_861_837_907, 1e-10);
            }
            other => panic!("expected scalar inv, got {other:?}"),
        }

        let upper = block_on(tcdf::tcdf_builtin(
            Value::Num(1.0e200),
            vec![Value::Num(1.0), Value::from("upper")],
        ))
        .unwrap();
        match upper {
            Value::Num(value) => assert_close(value / 1.0e-201, 3.183_098_861_837_907, 1e-12),
            other => panic!("expected scalar upper cdf, got {other:?}"),
        }
    }

    #[test]
    fn student_t_distribution_large_nu_uses_normal_limit() {
        let pdf = block_on(tpdf::tpdf_builtin(
            Value::Num(1.0),
            vec![Value::Num(1.0e12)],
        ))
        .unwrap();
        match pdf {
            Value::Num(value) => assert_close(value, INV_SQRT_2PI * (-0.5f64).exp(), 1e-12),
            other => panic!("expected scalar pdf, got {other:?}"),
        }

        let cdf = block_on(tcdf::tcdf_builtin(
            Value::Num(1.0),
            vec![Value::Num(1.0e12)],
        ))
        .unwrap();
        match cdf {
            Value::Num(value) => assert_close(value, 0.841_344_746_068_543, 1e-12),
            other => panic!("expected scalar cdf, got {other:?}"),
        }

        let inv = block_on(tinv::tinv_builtin(
            Value::Num(0.975),
            vec![Value::Num(1.0e12)],
        ))
        .unwrap();
        match inv {
            Value::Num(value) => assert_close(value, 1.959_963_984_540_053_8, 1e-12),
            other => panic!("expected scalar inv, got {other:?}"),
        }
    }

    #[test]
    fn binomial_chi_square_and_weibull_distribution_values() {
        let binomial = block_on(binocdf::binocdf_builtin(
            Value::Num(55.0),
            vec![Value::Num(100.0), Value::Num(0.5)],
        ))
        .unwrap();
        match binomial {
            Value::Num(value) => assert_close(value, 0.864_373_487_963_083, 1.0e-12),
            other => panic!("expected scalar binocdf, got {other:?}"),
        }

        let upper = block_on(binocdf::binocdf_builtin(
            Value::Num(55.0),
            vec![Value::Num(100.0), Value::Num(0.5), Value::from("upper")],
        ))
        .unwrap();
        match upper {
            Value::Num(value) => assert_close(value, 0.135_626_512_036_917, 1.0e-12),
            other => panic!("expected scalar upper binocdf, got {other:?}"),
        }

        let chi2 = block_on(chi2cdf::chi2cdf_builtin(
            Value::Num(3.0),
            vec![Value::Num(5.0)],
        ))
        .unwrap();
        match chi2 {
            Value::Num(value) => assert_close(value, 0.300_014_164_121_372, 1.0e-12),
            other => panic!("expected scalar chi2cdf, got {other:?}"),
        }

        let weibull = block_on(wblinv::wblinv_builtin(
            Value::Num(0.5),
            vec![Value::Num(3.0), Value::Num(4.0)],
        ))
        .unwrap();
        match weibull {
            Value::Num(value) => {
                assert_close(value, 3.0 * std::f64::consts::LN_2.powf(0.25), 1.0e-12)
            }
            other => panic!("expected scalar wblinv, got {other:?}"),
        }
    }

    #[test]
    fn distribution_helpers_broadcast_and_return_nan_for_bad_parameters() {
        let out = block_on(chi2cdf::chi2cdf_builtin(
            Value::Num(3.0),
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 2.0, 5.0], vec![1, 3]).unwrap(),
            )],
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_close(tensor.data[2], 0.300_014_164_121_372, 1.0e-12);
            }
            other => panic!("expected tensor chi2cdf, got {other:?}"),
        }

        let invalid = block_on(wblinv::wblinv_builtin(
            Value::Num(0.5),
            vec![Value::Num(-1.0), Value::Num(2.0)],
        ))
        .unwrap();
        assert!(matches!(invalid, Value::Num(value) if value.is_nan()));
    }

    #[test]
    fn binocdf_boundaries_and_invalid_parameters_match_distribution_contract() {
        let below = block_on(binocdf::binocdf_builtin(
            Value::Num(-1.0),
            vec![Value::Num(10.0), Value::Num(0.5)],
        ))
        .unwrap();
        assert_eq!(below, Value::Num(0.0));

        let above = block_on(binocdf::binocdf_builtin(
            Value::Num(10.0),
            vec![Value::Num(10.0), Value::Num(0.5)],
        ))
        .unwrap();
        assert_eq!(above, Value::Num(1.0));

        let always_zero = block_on(binocdf::binocdf_builtin(
            Value::Num(0.0),
            vec![Value::Num(10.0), Value::Num(0.0)],
        ))
        .unwrap();
        assert_eq!(always_zero, Value::Num(1.0));

        let impossible = block_on(binocdf::binocdf_builtin(
            Value::Num(5.0),
            vec![Value::Num(10.5), Value::Num(0.5)],
        ))
        .unwrap();
        assert!(matches!(impossible, Value::Num(value) if value.is_nan()));
    }

    #[test]
    fn wblinv_preserves_tiny_positive_probabilities() {
        let out = block_on(wblinv::wblinv_builtin(
            Value::Num(1.0e-20),
            vec![Value::Num(3.0), Value::Num(4.0)],
        ))
        .unwrap();
        match out {
            Value::Num(value) => {
                assert!(value > 0.0);
                assert_close(value, 3.0e-5, 1.0e-16);
            }
            other => panic!("expected scalar wblinv, got {other:?}"),
        }
    }
}
