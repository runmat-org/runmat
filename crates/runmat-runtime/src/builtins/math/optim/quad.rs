//! MATLAB-compatible legacy `quad` builtin for finite scalar quadrature.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    LogicalArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::optim::common::{call_function, value_to_scalar};
use crate::builtins::math::optim::type_resolvers::numerical_integral_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "quad";
const DEFAULT_TOL: f64 = 1.0e-6;
const MAX_DEPTH: usize = 30;
const MAX_FUN_EVALS: usize = 100_000;

const QUAD_OUTPUT_Q: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "q",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numerical integral estimate.",
}];

const QUAD_OUTPUT_Q_FCNT: [BuiltinParamDescriptor; 2] = [
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
        description: "Number of integrand evaluations.",
    },
];

const QUAD_INPUTS_CORE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar integrand callback.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower integration bound.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper integration bound.",
    },
];

const QUAD_INPUTS_TOL_TRACE_ARGS: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar integrand callback.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower integration bound.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper integration bound.",
    },
    BuiltinParamDescriptor {
        name: "tol",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1e-6"),
        description: "Absolute error tolerance. Empty uses the default.",
    },
    BuiltinParamDescriptor {
        name: "trace",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("false"),
        description: "Nonzero value prints legacy [fcnEvals, a, b-a, Q] trace rows.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional arguments forwarded to the integrand.",
    },
];

const QUAD_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "q = quad(fun, a, b)",
        inputs: &QUAD_INPUTS_CORE,
        outputs: &QUAD_OUTPUT_Q,
    },
    BuiltinSignatureDescriptor {
        label: "q = quad(fun, a, b, tol, trace, p1, p2, ...)",
        inputs: &QUAD_INPUTS_TOL_TRACE_ARGS,
        outputs: &QUAD_OUTPUT_Q,
    },
    BuiltinSignatureDescriptor {
        label: "[q, fcnt] = quad(fun, a, b)",
        inputs: &QUAD_INPUTS_CORE,
        outputs: &QUAD_OUTPUT_Q_FCNT,
    },
    BuiltinSignatureDescriptor {
        label: "[q, fcnt] = quad(fun, a, b, tol, trace, p1, p2, ...)",
        inputs: &QUAD_INPUTS_TOL_TRACE_ARGS,
        outputs: &QUAD_OUTPUT_Q_FCNT,
    },
];

const QUAD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUAD.INVALID_ARGUMENT",
    identifier: Some("RunMat:quad:InvalidArgument"),
    when: "Tolerance, trace flag, or argument grammar is invalid.",
    message: "quad: invalid argument",
};

const QUAD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUAD.INVALID_INPUT",
    identifier: Some("RunMat:quad:InvalidInput"),
    when: "Bounds, integrand values, or adaptive solver semantics are invalid.",
    message: "quad: invalid input",
};

const QUAD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUAD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:quad:TooManyOutputs"),
    when: "`quad` is called with more than two requested output arguments.",
    message: "quad: too many output arguments",
};

const QUAD_ERRORS: [BuiltinErrorDescriptor; 3] = [
    QUAD_ERROR_INVALID_ARGUMENT,
    QUAD_ERROR_INVALID_INPUT,
    QUAD_ERROR_TOO_MANY_OUTPUTS,
];

pub const QUAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &QUAD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &QUAD_ERRORS,
};

fn quad_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("quad:") {
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

fn quad_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        quad_error_with_detail(fallback, err.message())
    }
}

fn validate_requested_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 2) {
        return Err(quad_error_with_detail(
            &QUAD_ERROR_TOO_MANY_OUTPUTS,
            "quad: too many output arguments; maximum is 2",
        ));
    }
    Ok(())
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::quad")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "quad",
    op_kind: GpuOpKind::Custom("legacy-adaptive-simpson"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host adaptive Simpson solver. Callback computations may use GPU-aware builtins, but the adaptive integration loop runs on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::quad")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "quad",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Legacy adaptive quadrature repeatedly invokes user code and terminates fusion planning.",
};

#[runtime_builtin(
    name = "quad",
    category = "math/optim",
    summary = "Approximate finite scalar definite integrals using legacy adaptive Simpson quadrature.",
    keywords = "quad,numerical integration,adaptive simpson,quadrature,function handle",
    accel = "sink",
    type_resolver(numerical_integral_type),
    descriptor(crate::builtins::math::optim::quad::QUAD_DESCRIPTOR),
    builtin_path = "crate::builtins::math::optim::quad"
)]
async fn quad_builtin(
    function: Value,
    a: Value,
    b: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    validate_requested_outputs()?;
    let options = QuadOptions::parse(rest)
        .await
        .map_err(|err| quad_map_error(err, &QUAD_ERROR_INVALID_ARGUMENT))?;
    let a = scalar_real("lower bound", a)
        .await
        .map_err(|err| quad_map_error(err, &QUAD_ERROR_INVALID_INPUT))?;
    let b = scalar_real("upper bound", b)
        .await
        .map_err(|err| quad_map_error(err, &QUAD_ERROR_INVALID_INPUT))?;

    let result = if a == b {
        QuadResult {
            q: 0.0,
            func_count: 0,
        }
    } else {
        let sign = if b < a { -1.0 } else { 1.0 };
        let lo = a.min(b);
        let hi = a.max(b);
        let mut result = integrate_quad(&function, lo, hi, &options)
            .await
            .map_err(|err| quad_map_error(err, &QUAD_ERROR_INVALID_INPUT))?;
        result.q *= sign;
        result
    };

    finalize(result)
}

struct QuadOptions {
    tol: f64,
    trace: bool,
    extra_args: Vec<Value>,
}

impl QuadOptions {
    async fn parse(rest: Vec<Value>) -> BuiltinResult<Self> {
        let mut values = rest.into_iter();
        let tol = match values.next() {
            Some(value) => parse_optional_tol(value).await?,
            None => DEFAULT_TOL,
        };
        let trace = match values.next() {
            Some(value) => parse_optional_trace(value).await?,
            None => false,
        };
        Ok(Self {
            tol,
            trace,
            extra_args: values.collect(),
        })
    }
}

async fn parse_optional_tol(value: Value) -> BuiltinResult<f64> {
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    if is_empty_value(&value) {
        return Ok(DEFAULT_TOL);
    }
    let tol = scalar_real_sync("tolerance", value, &QUAD_ERROR_INVALID_ARGUMENT)?;
    if tol > 0.0 {
        Ok(tol)
    } else {
        Err(quad_error_with_detail(
            &QUAD_ERROR_INVALID_ARGUMENT,
            "tolerance must be a positive finite scalar",
        ))
    }
}

async fn parse_optional_trace(value: Value) -> BuiltinResult<bool> {
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    if is_empty_value(&value) {
        return Ok(false);
    }
    Ok(scalar_real_sync("trace", value, &QUAD_ERROR_INVALID_ARGUMENT)? != 0.0)
}

fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Tensor(Tensor { data, .. }) => data.is_empty(),
        Value::LogicalArray(LogicalArray { data, .. }) => data.is_empty(),
        _ => false,
    }
}

async fn scalar_real(label: &str, value: Value) -> BuiltinResult<f64> {
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    scalar_real_sync(label, value, &QUAD_ERROR_INVALID_INPUT)
}

fn scalar_real_sync(
    label: &str,
    value: Value,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<f64> {
    let parsed = match value {
        Value::Num(n) => n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(flag) => {
            if flag {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(Tensor { data, .. }) if data.len() == 1 => data[0],
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            if data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(quad_error_with_detail(
                error,
                format!("{label} must be a finite real scalar, got {other:?}"),
            ))
        }
    };
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(quad_error_with_detail(
            error,
            format!("{label} must be finite"),
        ))
    }
}

#[derive(Clone, Copy)]
struct QuadResult {
    q: f64,
    func_count: usize,
}

async fn integrate_quad(
    function: &Value,
    a: f64,
    b: f64,
    options: &QuadOptions,
) -> BuiltinResult<QuadResult> {
    let fa = call_integrand(function, a, &options.extra_args).await?;
    let c = midpoint(a, b);
    let fc = call_integrand(function, c, &options.extra_args).await?;
    let fb = call_integrand(function, b, &options.extra_args).await?;
    let whole = simpson(a, b, fa, fc, fb);
    let mut func_count = 3usize;
    let mut trace = options.trace.then_some(QuadTrace);
    let q = adaptive_simpson(
        function,
        &options.extra_args,
        SimpsonState {
            a,
            b,
            fa,
            fc,
            fb,
            whole,
            tol: options.tol,
            depth: MAX_DEPTH,
        },
        &mut func_count,
        &mut trace,
    )
    .await?;
    Ok(QuadResult { q, func_count })
}

#[derive(Clone, Copy)]
struct SimpsonState {
    a: f64,
    b: f64,
    fa: f64,
    fc: f64,
    fb: f64,
    whole: f64,
    tol: f64,
    depth: usize,
}

#[async_recursion::async_recursion(?Send)]
async fn adaptive_simpson(
    function: &Value,
    extra_args: &[Value],
    state: SimpsonState,
    func_count: &mut usize,
    trace: &mut Option<QuadTrace>,
) -> BuiltinResult<f64> {
    if *func_count + 2 > MAX_FUN_EVALS {
        return Err(quad_error_with_detail(
            &QUAD_ERROR_INVALID_INPUT,
            "exceeded maximum function evaluations",
        ));
    }

    let c = midpoint(state.a, state.b);
    let d = midpoint(state.a, c);
    let e = midpoint(c, state.b);
    let fd = call_integrand(function, d, extra_args).await?;
    let fe = call_integrand(function, e, extra_args).await?;
    *func_count += 2;

    let left = simpson(state.a, c, state.fa, fd, state.fc);
    let right = simpson(c, state.b, state.fc, fe, state.fb);
    let refined = left + right;
    let error = refined - state.whole;
    if let Some(trace) = trace {
        trace.record(*func_count, state.a, state.b, refined, error);
    }
    if error.abs() <= 15.0 * state.tol {
        return Ok(refined + error / 15.0);
    }
    if state.depth == 0 {
        return Err(quad_error_with_detail(
            &QUAD_ERROR_INVALID_INPUT,
            "adaptive Simpson quadrature did not converge",
        ));
    }

    let left_value = adaptive_simpson(
        function,
        extra_args,
        SimpsonState {
            a: state.a,
            b: c,
            fa: state.fa,
            fc: fd,
            fb: state.fc,
            whole: left,
            tol: state.tol * 0.5,
            depth: state.depth - 1,
        },
        func_count,
        trace,
    )
    .await?;
    let right_value = adaptive_simpson(
        function,
        extra_args,
        SimpsonState {
            a: c,
            b: state.b,
            fa: state.fc,
            fc: fe,
            fb: state.fb,
            whole: right,
            tol: state.tol * 0.5,
            depth: state.depth - 1,
        },
        func_count,
        trace,
    )
    .await?;
    Ok(left_value + right_value)
}

fn midpoint(a: f64, b: f64) -> f64 {
    a + (b - a) * 0.5
}

fn simpson(a: f64, b: f64, fa: f64, fm: f64, fb: f64) -> f64 {
    (b - a) * (fa + 4.0 * fm + fb) / 6.0
}

async fn call_integrand(function: &Value, x: f64, extra_args: &[Value]) -> BuiltinResult<f64> {
    let mut args = Vec::with_capacity(1 + extra_args.len());
    args.push(Value::Num(x));
    args.extend(extra_args.iter().cloned());
    let value = call_function(function, args).await?;
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    value_to_scalar(NAME, value)
}

struct QuadTrace;

impl QuadTrace {
    fn record(&mut self, func_count: usize, a: f64, b: f64, q: f64, _err: f64) {
        crate::console::record_console_line(
            crate::console::ConsoleStream::Stdout,
            format!(
                "    {func_count:>5}    {a:13.6e} {width:13.6e} {q:13.6e}",
                width = b - a,
            ),
        );
    }
}

fn finalize(result: QuadResult) -> BuiltinResult<Value> {
    let q = Value::Num(result.q);
    let fcnt = Value::Num(result.func_count as f64);
    match crate::output_count::current_output_count() {
        None => Ok(q),
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(crate::output_count::output_list_with_padding(1, vec![q])),
        Some(2) => Ok(crate::output_count::output_list_with_padding(
            2,
            vec![q, fcnt],
        )),
        Some(_) => Err(quad_error_with_detail(
            &QUAD_ERROR_TOO_MANY_OUTPUTS,
            "quad: too many output arguments; maximum is 2",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::sync::Arc;

    #[test]
    fn quad_integrates_sine_with_default_tolerance() {
        let result = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(std::f64::consts::PI),
            Vec::new(),
        ))
        .expect("quad");
        match result {
            Value::Num(value) => assert!((value - 2.0).abs() < 1.0e-6),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn quad_respects_tighter_tolerance_on_polynomial() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, requested_outputs| {
                assert_eq!(requested_outputs, 1);
                let x = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected x, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(x * x)) })
            },
        )));

        let result = block_on(quad_builtin(
            Value::BoundFunctionHandle {
                name: "square".to_string(),
                function: 7,
            },
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Num(1.0e-10)],
        ))
        .expect("quad");
        match result {
            Value::Num(value) => assert!((value - (1.0 / 3.0)).abs() < 1.0e-10),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn quad_two_outputs_include_function_count() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(std::f64::consts::PI),
            Vec::new(),
        ))
        .expect("quad");
        match result {
            Value::OutputList(outputs) => {
                assert_eq!(outputs.len(), 2);
                assert!(matches!(&outputs[0], Value::Num(value) if (value - 2.0).abs() < 1.0e-6));
                assert!(matches!(&outputs[1], Value::Num(fcnt) if *fcnt >= 5.0));
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn quad_trace_records_rows() {
        crate::console::reset_thread_buffer();
        let result = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(std::f64::consts::PI),
            vec![Value::Num(1.0e-6), Value::Num(1.0)],
        ))
        .expect("quad");
        assert!(matches!(result, Value::Num(_)));

        let joined = crate::console::take_thread_buffer()
            .into_iter()
            .map(|entry| entry.text)
            .collect::<String>();
        let first_row: Vec<&str> = joined
            .lines()
            .next()
            .expect("expected at least one trace row")
            .split_whitespace()
            .collect();
        assert_eq!(first_row.len(), 4, "{joined}");
        assert_eq!(first_row[0], "5", "{joined}");
        assert!((first_row[1].parse::<f64>().unwrap() - 0.0).abs() < 1.0e-12);
        assert!(
            (first_row[2].parse::<f64>().unwrap() - std::f64::consts::PI).abs() < 1.0e-6,
            "{joined}"
        );
    }

    #[test]
    fn quad_forwards_extra_arguments_after_tol_and_trace() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 42);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args.len(), 2);
                let x = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected x, got {other:?}"),
                };
                let scale = match &args[1] {
                    Value::Num(value) => *value,
                    other => panic!("expected scale, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(scale * x)) })
            },
        )));

        let result = block_on(quad_builtin(
            Value::BoundFunctionHandle {
                name: "scaled_line".to_string(),
                function: 42,
            },
            Value::Num(0.0),
            Value::Num(2.0),
            vec![
                Value::Tensor(Tensor::zeros(vec![0, 0])),
                Value::Tensor(Tensor::zeros(vec![0, 0])),
                Value::Num(3.0),
            ],
        ))
        .expect("quad");
        match result {
            Value::Num(value) => assert!((value - 6.0).abs() < 1.0e-8),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn quad_handles_oscillatory_integrand() {
        let result = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(2.0 * std::f64::consts::PI),
            vec![Value::Num(1.0e-8)],
        ))
        .expect("quad");
        match result {
            Value::Num(value) => assert!(value.abs() < 1.0e-8),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn quad_handles_integrable_endpoint_shape() {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let x = match &args[0] {
                    Value::Num(value) => *value,
                    other => panic!("expected x, got {other:?}"),
                };
                Box::pin(async move { Ok(Value::Num(x.sqrt())) })
            },
        )));

        let result = block_on(quad_builtin(
            Value::BoundFunctionHandle {
                name: "sqrt_fn".to_string(),
                function: 9,
            },
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Num(1.0e-7)],
        ))
        .expect("quad");
        match result {
            Value::Num(value) => assert!((value - (2.0 / 3.0)).abs() < 1.0e-6),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn quad_reversed_bounds_negate_result() {
        let result = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(std::f64::consts::PI),
            Value::Num(0.0),
            Vec::new(),
        ))
        .expect("quad");
        match result {
            Value::Num(value) => assert!((value + 2.0).abs() < 1.0e-6),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn quad_rejects_more_than_two_outputs() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let err = block_on(quad_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(1.0),
            Vec::new(),
        ))
        .expect_err("too many outputs should fail");
        assert_eq!(err.identifier(), Some("RunMat:quad:TooManyOutputs"));
    }

    #[test]
    fn quad_descriptor_signatures_cover_legacy_forms() {
        let labels: Vec<&str> = QUAD_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "q = quad(fun, a, b)",
                "q = quad(fun, a, b, tol, trace, p1, p2, ...)",
                "[q, fcnt] = quad(fun, a, b)",
                "[q, fcnt] = quad(fun, a, b, tol, trace, p1, p2, ...)",
            ]
        );

        let codes: Vec<&str> = QUAD_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec![
                "RM.QUAD.INVALID_ARGUMENT",
                "RM.QUAD.INVALID_INPUT",
                "RM.QUAD.TOO_MANY_OUTPUTS",
            ]
        );
    }
}
