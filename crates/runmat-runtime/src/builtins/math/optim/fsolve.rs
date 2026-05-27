//! MATLAB-compatible `fsolve` builtin for nonlinear systems.

use nalgebra::{DMatrix, DVector};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::optim::common::{
    call_function, initial_guess, option_f64, option_string, option_usize, value_to_real_vector,
    vector_to_value,
};
use crate::builtins::math::optim::type_resolvers::nonlinear_solve_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "fsolve";
const DEFAULT_TOL_X: f64 = 1.0e-6;
const DEFAULT_TOL_FUN: f64 = 1.0e-6;
const DEFAULT_MAX_ITER: usize = 400;

const FSOLVE_OUTPUT_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Approximate solution vector/scalar.",
}];

const FSOLVE_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "System residual callback.",
    },
    BuiltinParamDescriptor {
        name: "x0",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Initial guess scalar/vector.",
    },
];

const FSOLVE_INPUTS_WITH_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "System residual callback.",
    },
    BuiltinParamDescriptor {
        name: "x0",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Initial guess scalar/vector.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Options struct from optimset.",
    },
];

const FSOLVE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "x = fsolve(fun, x0)",
        inputs: &FSOLVE_INPUTS_CORE,
        outputs: &FSOLVE_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = fsolve(fun, x0, options)",
        inputs: &FSOLVE_INPUTS_WITH_OPTIONS,
        outputs: &FSOLVE_OUTPUT_X,
    },
];

const FSOLVE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSOLVE.INVALID_ARGUMENT",
    identifier: Some("RunMat:fsolve:InvalidArgument"),
    when: "Argument grammar/options configuration is invalid.",
    message: "fsolve: invalid argument",
};

const FSOLVE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSOLVE.INVALID_INPUT",
    identifier: Some("RunMat:fsolve:InvalidInput"),
    when: "Initial guess/callback/iteration semantics are invalid.",
    message: "fsolve: invalid input",
};

const FSOLVE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [FSOLVE_ERROR_INVALID_ARGUMENT, FSOLVE_ERROR_INVALID_INPUT];

pub const FSOLVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FSOLVE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FSOLVE_ERRORS,
};

fn fsolve_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("fsolve:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fsolve_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        fsolve_error_with_detail(fallback, err.message())
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::fsolve")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fsolve",
    op_kind: GpuOpKind::Custom("nonlinear-solve"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host finite-difference Levenberg-Marquardt solver. Callback computations may use GPU-aware builtins.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::fsolve")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fsolve",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Nonlinear solving repeatedly invokes user code and terminates fusion planning.",
};

#[runtime_builtin(
    name = "fsolve",
    category = "math/optim",
    summary = "Solve a scalar or vector nonlinear equation system using finite-difference Levenberg-Marquardt iterations.",
    keywords = "fsolve,nonlinear solve,root finding,levenberg-marquardt,jacobian",
    accel = "sink",
    type_resolver(nonlinear_solve_type),
    descriptor(crate::builtins::math::optim::fsolve::FSOLVE_DESCRIPTOR),
    builtin_path = "crate::builtins::math::optim::fsolve"
)]
async fn fsolve_builtin(function: Value, x0: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(fsolve_error_with_detail(
            &FSOLVE_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    let options = parse_options(rest.first())
        .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_ARGUMENT))?;
    let opts = FsolveOptions::from_struct(options.as_ref())
        .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_ARGUMENT))?;
    let guess = initial_guess(NAME, x0)
        .await
        .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_INPUT))?;
    let solution = solve(&function, guess.values, &guess.shape, guess.scalar, &opts)
        .await
        .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_INPUT))?;
    vector_to_value(NAME, solution, &guess.shape, guess.scalar)
        .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_INPUT))
}

fn parse_options(value: Option<&Value>) -> BuiltinResult<Option<StructValue>> {
    match value {
        None => Ok(None),
        Some(Value::Struct(options)) => Ok(Some(options.clone())),
        Some(other) => Err(fsolve_error_with_detail(
            &FSOLVE_ERROR_INVALID_ARGUMENT,
            format!("options must be a struct, got {other:?}"),
        )),
    }
}

#[derive(Clone, Copy)]
struct FsolveOptions {
    tol_x: f64,
    tol_fun: f64,
    max_iter: usize,
    max_fun_evals: usize,
}

impl FsolveOptions {
    fn from_struct(options: Option<&StructValue>) -> BuiltinResult<Self> {
        let display = option_string(options, "Display", "off")?;
        if !matches!(display.as_str(), "off" | "none" | "final" | "iter") {
            return Err(fsolve_error_with_detail(
                &FSOLVE_ERROR_INVALID_ARGUMENT,
                "option Display must be 'off', 'none', 'final', or 'iter'",
            ));
        }
        let tol_x = option_f64(NAME, options, "TolX", DEFAULT_TOL_X)?;
        let tol_fun = option_f64(NAME, options, "TolFun", DEFAULT_TOL_FUN)?;
        if tol_x <= 0.0 || tol_fun <= 0.0 {
            return Err(fsolve_error_with_detail(
                &FSOLVE_ERROR_INVALID_ARGUMENT,
                "options TolX and TolFun must be positive",
            ));
        }
        let max_iter = option_usize(NAME, options, "MaxIter", DEFAULT_MAX_ITER)?.max(1);
        let max_fun_evals = option_usize(NAME, options, "MaxFunEvals", 100 * max_iter)?.max(1);
        Ok(Self {
            tol_x,
            tol_fun,
            max_iter,
            max_fun_evals,
        })
    }
}

async fn solve(
    function: &Value,
    mut x: Vec<f64>,
    shape: &[usize],
    scalar: bool,
    options: &FsolveOptions,
) -> BuiltinResult<Vec<f64>> {
    let n = x.len();
    if n == 0 {
        return Err(fsolve_error_with_detail(
            &FSOLVE_ERROR_INVALID_INPUT,
            "initial guess cannot be empty",
        ));
    }

    let mut residual = eval_residual(function, &x, shape, scalar).await?;
    let mut evals = 1usize;
    let mut lambda = 1.0e-3;

    if residual_norm_inf(&residual) <= options.tol_fun {
        return Ok(x);
    }

    for _ in 0..options.max_iter {
        if evals >= options.max_fun_evals {
            return Err(fsolve_error_with_detail(
                &FSOLVE_ERROR_INVALID_INPUT,
                "exceeded maximum function evaluations",
            ));
        }
        let jacobian =
            finite_difference_jacobian(function, &x, shape, scalar, &residual, &mut evals, options)
                .await?;
        let j = DMatrix::from_row_slice(residual.len(), n, &jacobian);
        let f = DVector::from_column_slice(&residual);
        let gradient = j.transpose() * &f;
        let mut accepted = false;

        for _ in 0..8 {
            let normal = j.transpose() * &j + DMatrix::<f64>::identity(n, n) * lambda;
            let rhs = -&gradient;
            let Some(delta) = normal.lu().solve(&rhs) else {
                lambda *= 10.0;
                continue;
            };
            let trial = x
                .iter()
                .zip(delta.iter())
                .map(|(xi, di)| xi + di)
                .collect::<Vec<_>>();
            let trial_residual = eval_residual(function, &trial, shape, scalar).await?;
            evals += 1;

            if norm2(&trial_residual) < norm2(&residual) {
                let step_norm = delta
                    .iter()
                    .fold(0.0_f64, |acc, value| acc.max(value.abs()));
                let x_norm = x.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
                x = trial;
                residual = trial_residual;
                lambda = (lambda * 0.3).max(1.0e-12);
                accepted = true;
                if residual_norm_inf(&residual) <= options.tol_fun
                    || step_norm <= options.tol_x * (1.0 + x_norm)
                {
                    return Ok(x);
                }
                break;
            }

            lambda *= 10.0;
            if evals >= options.max_fun_evals {
                return Err(fsolve_error_with_detail(
                    &FSOLVE_ERROR_INVALID_INPUT,
                    "exceeded maximum function evaluations",
                ));
            }
        }

        if !accepted {
            return Err(fsolve_error_with_detail(
                &FSOLVE_ERROR_INVALID_INPUT,
                "iteration stalled before convergence",
            ));
        }
    }

    Err(fsolve_error_with_detail(
        &FSOLVE_ERROR_INVALID_INPUT,
        "exceeded maximum iterations",
    ))
}

async fn eval_residual(
    function: &Value,
    x: &[f64],
    shape: &[usize],
    scalar: bool,
) -> BuiltinResult<Vec<f64>> {
    let arg = if scalar {
        Value::Num(x[0])
    } else {
        Value::Tensor(
            runmat_builtins::Tensor::new(x.to_vec(), shape.to_vec())
                .map_err(|e| fsolve_error_with_detail(&FSOLVE_ERROR_INVALID_INPUT, e))?,
        )
    };
    let value = call_function(function, vec![arg]).await?;
    let residual = value_to_real_vector(NAME, value).await?;
    if residual.is_empty() {
        Err(fsolve_error_with_detail(
            &FSOLVE_ERROR_INVALID_INPUT,
            "function value must not be empty",
        ))
    } else {
        Ok(residual)
    }
}

async fn finite_difference_jacobian(
    function: &Value,
    x: &[f64],
    shape: &[usize],
    scalar: bool,
    residual: &[f64],
    evals: &mut usize,
    options: &FsolveOptions,
) -> BuiltinResult<Vec<f64>> {
    let m = residual.len();
    let n = x.len();
    let mut jacobian = vec![0.0; m * n];

    for col in 0..n {
        if *evals >= options.max_fun_evals {
            return Err(fsolve_error_with_detail(
                &FSOLVE_ERROR_INVALID_INPUT,
                "exceeded maximum function evaluations",
            ));
        }
        let mut perturbed = x.to_vec();
        let step = f64::EPSILON.sqrt() * (x[col].abs() + 1.0);
        perturbed[col] += step;
        let next = eval_residual(function, &perturbed, shape, scalar).await?;
        *evals += 1;
        if next.len() != m {
            return Err(fsolve_error_with_detail(
                &FSOLVE_ERROR_INVALID_INPUT,
                "function output size changed during finite differencing",
            ));
        }
        for row in 0..m {
            jacobian[row * n + col] = (next[row] - residual[row]) / step;
        }
    }

    Ok(jacobian)
}

fn norm2(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn residual_norm_inf(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;
    use std::sync::{Arc, Mutex};

    #[test]
    fn fsolve_scalar_builtin_handle() {
        let root = block_on(fsolve_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(3.0),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Num(n) => assert!((n - std::f64::consts::PI).abs() < 1.0e-5),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fsolve_vector_system_via_semantic_resolver() {
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_name| {
                Some(0)
            })));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
            std::sync::Arc::new(|_function, args, _requested_outputs| {
                let x = match &args[0] {
                    Value::Tensor(t) => t.data.clone(),
                    _ => panic!("expected tensor input"),
                };
                Box::pin(async move {
                    Ok(Value::Tensor(
                        Tensor::new(
                            vec![x[0] * x[0] + x[1] * x[1] - 4.0, x[0] * x[1] - 1.0],
                            vec![2, 1],
                        )
                        .unwrap(),
                    ))
                })
            }),
        ));
        let x0 = Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap();
        let root = block_on(fsolve_builtin(
            Value::FunctionHandle("system".into()),
            Value::Tensor(x0),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Tensor(t) => {
                assert!((t.data[0] * t.data[0] + t.data[1] * t.data[1] - 4.0).abs() < 1.0e-5);
                assert!((t.data[0] * t.data[1] - 1.0).abs() < 1.0e-5);
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fsolve_preserves_row_vector_shape_for_callback() {
        let seen_shapes = Arc::new(Mutex::new(Vec::new()));
        let seen_shapes_for_invoker = Arc::clone(&seen_shapes);
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_name| {
                Some(0)
            })));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_function, args, _requested_outputs| {
                let (x, shape) = match &args[0] {
                    Value::Tensor(t) => (t.data.clone(), t.shape.clone()),
                    other => panic!("expected tensor input, got {other:?}"),
                };
                assert_eq!(shape, vec![1, 2]);
                seen_shapes_for_invoker.lock().unwrap().push(shape.clone());
                Box::pin(async move {
                    Ok(Value::Tensor(
                        Tensor::new(vec![x[0] - 3.0, x[1] - 4.0], shape).unwrap(),
                    ))
                })
            },
        )));
        let x0 = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
        let root = block_on(fsolve_builtin(
            Value::FunctionHandle("row_system".into()),
            Value::Tensor(x0),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.data[0] - 3.0).abs() < 1.0e-5);
                assert!((t.data[1] - 4.0).abs() < 1.0e-5);
            }
            other => panic!("unexpected value {other:?}"),
        }
        assert!(!seen_shapes.lock().unwrap().is_empty());
    }

    #[test]
    fn fsolve_preserves_matrix_shape_for_callback() {
        let seen_shapes = Arc::new(Mutex::new(Vec::new()));
        let seen_shapes_for_invoker = Arc::clone(&seen_shapes);
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_name| {
                Some(0)
            })));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_function, args, _requested_outputs| {
                let (x, shape) = match &args[0] {
                    Value::Tensor(t) => (t.data.clone(), t.shape.clone()),
                    other => panic!("expected tensor input, got {other:?}"),
                };
                assert_eq!(shape, vec![2, 2]);
                seen_shapes_for_invoker.lock().unwrap().push(shape.clone());
                Box::pin(async move {
                    Ok(Value::Tensor(
                        Tensor::new(vec![x[0] - 1.0, x[1] - 2.0, x[2] - 3.0, x[3] - 4.0], shape)
                            .unwrap(),
                    ))
                })
            },
        )));
        let x0 = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let root = block_on(fsolve_builtin(
            Value::FunctionHandle("matrix_system".into()),
            Value::Tensor(x0),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!((t.data[0] - 1.0).abs() < 1.0e-5);
                assert!((t.data[1] - 2.0).abs() < 1.0e-5);
                assert!((t.data[2] - 3.0).abs() < 1.0e-5);
                assert!((t.data[3] - 4.0).abs() < 1.0e-5);
            }
            other => panic!("unexpected value {other:?}"),
        }
        assert!(!seen_shapes.lock().unwrap().is_empty());
    }

    #[test]
    fn fsolve_accepts_semantic_function_handle_callback() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 43);
                assert_eq!(requested_outputs, 1);
                let x = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected scalar numeric argument, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(x - 3.0)) })
            },
        )));
        let root = block_on(fsolve_builtin(
            Value::BoundFunctionHandle {
                name: "system_function".to_string(),
                function: 43,
            },
            Value::Num(1.0),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Num(n) => assert!((n - 3.0).abs() < 1.0e-5),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fsolve_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FSOLVE_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec!["x = fsolve(fun, x0)", "x = fsolve(fun, x0, options)"]
        );

        let codes: Vec<&str> = FSOLVE_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec!["RM.FSOLVE.INVALID_ARGUMENT", "RM.FSOLVE.INVALID_INPUT"]
        );
    }

    #[test]
    fn fsolve_too_many_args_uses_stable_identifier() {
        let err = block_on(fsolve_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(1.0),
            vec![
                Value::Struct(StructValue::new()),
                Value::Struct(StructValue::new()),
            ],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:fsolve:InvalidArgument"));
    }
}
