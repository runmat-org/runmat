//! Normal distribution compatibility helpers.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor;
use crate::builtins::math::elementwise::erfcinv::erfcinv_scalar;
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

const INPUTS_X: [BuiltinParamDescriptor; 1] = [INPUT_X];
const INPUTS_X_MU: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_MU];
const INPUTS_X_MU_SIGMA: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_MU, INPUT_SIGMA];
const INPUTS_P: [BuiltinParamDescriptor; 1] = [INPUT_P];
const INPUTS_P_MU: [BuiltinParamDescriptor; 2] = [INPUT_P, INPUT_MU];
const INPUTS_P_MU_SIGMA: [BuiltinParamDescriptor; 3] = [INPUT_P, INPUT_MU, INPUT_SIGMA];

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

fn normal_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
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
    let (x, mu, shape) = broadcast_pair(name, &x, &mu)?;
    let (sigma, _, shape2) = broadcast_pair(
        name,
        &sigma,
        &Tensor::new(vec![0.0; x.len()], shape.clone()).unwrap(),
    )?;
    if shape2 != shape {
        return Err(normal_error(
            name,
            format!("{name}: operands must have compatible sizes"),
        ));
    }
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

fn broadcast_pair(
    name: &str,
    lhs: &Tensor,
    rhs: &Tensor,
) -> BuiltinResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    tensor::binary_numeric_tensors(lhs, rhs, name, name)
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
}
