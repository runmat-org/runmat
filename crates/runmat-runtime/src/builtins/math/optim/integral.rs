//! MATLAB-compatible `integral` builtin for finite scalar numerical integration.

use runmat_builtins::{LogicalArray, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::optim::common::{call_function, optim_error};
use crate::builtins::math::optim::type_resolvers::numerical_integral_type;
use crate::BuiltinResult;

const NAME: &str = "integral";
const DEFAULT_ABS_TOL: f64 = 1.0e-10;
const DEFAULT_REL_TOL: f64 = 1.0e-6;
const DEFAULT_MAX_FUN_EVALS: usize = 10_000;
const MAX_DEPTH: usize = 30;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::integral")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "integral",
    op_kind: GpuOpKind::Custom("adaptive-quadrature"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host adaptive quadrature solver. Callback computations may use GPU-aware builtins, but the adaptive integration loop runs on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::integral")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "integral",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Adaptive integration repeatedly invokes user code and terminates fusion planning.",
};

#[runtime_builtin(
    name = "integral",
    category = "math/optim",
    summary = "Approximate a finite scalar definite integral using adaptive quadrature.",
    keywords = "integral,numerical integration,adaptive quadrature,quadrature,function handle",
    accel = "sink",
    type_resolver(numerical_integral_type),
    builtin_path = "crate::builtins::math::optim::integral"
)]
async fn integral_builtin(
    function: Value,
    a: Value,
    b: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let options = IntegralOptions::parse(rest)?;
    let a = scalar_bound("lower bound", a).await?;
    let b = scalar_bound("upper bound", b).await?;
    if a == b {
        return Ok(Value::Num(0.0));
    }

    let sign = if b < a { -1.0 } else { 1.0 };
    let lo = a.min(b);
    let hi = a.max(b);
    let result = integrate_finite_scalar(&function, lo, hi, &options).await?;
    Ok(Value::Num(sign * result))
}

#[derive(Clone, Copy)]
struct IntegralOptions {
    abs_tol: f64,
    rel_tol: f64,
    max_fun_evals: usize,
}

impl IntegralOptions {
    fn parse(rest: Vec<Value>) -> BuiltinResult<Self> {
        let mut options = Self {
            abs_tol: DEFAULT_ABS_TOL,
            rel_tol: DEFAULT_REL_TOL,
            max_fun_evals: DEFAULT_MAX_FUN_EVALS,
        };
        if rest.is_empty() {
            return Ok(options);
        }
        if rest.len() == 1 {
            return match &rest[0] {
                Value::Struct(fields) => {
                    options.apply_struct(fields)?;
                    Ok(options)
                }
                other => Err(optim_error(
                    NAME,
                    format!("integral: expected option name/value pairs, got {other:?}"),
                )),
            };
        }
        if !rest.len().is_multiple_of(2) {
            return Err(optim_error(
                NAME,
                "integral: expected option name/value pairs",
            ));
        }
        for pair in rest.chunks(2) {
            let name = option_name(&pair[0])?;
            options.apply_option(&name, &pair[1])?;
        }
        options.validate()?;
        Ok(options)
    }

    fn apply_struct(&mut self, fields: &StructValue) -> BuiltinResult<()> {
        for (name, value) in &fields.fields {
            self.apply_option(name, value)?;
        }
        self.validate()
    }

    fn apply_option(&mut self, name: &str, value: &Value) -> BuiltinResult<()> {
        match name.to_ascii_lowercase().as_str() {
            "abstol" => self.abs_tol = numeric_option("AbsTol", value)?,
            "reltol" => self.rel_tol = numeric_option("RelTol", value)?,
            "maxfunevals" | "maxintervalcount" => {
                self.max_fun_evals = integer_option(name, value)?.max(3)
            }
            "arrayvalued" => {
                if bool_option("ArrayValued", value)? {
                    return Err(optim_error(
                        NAME,
                        "integral: ArrayValued true is not supported yet",
                    ));
                }
            }
            other => {
                return Err(optim_error(
                    NAME,
                    format!("integral: unsupported option {other}"),
                ))
            }
        }
        Ok(())
    }

    fn validate(&self) -> BuiltinResult<()> {
        if self.abs_tol < 0.0 {
            return Err(optim_error(NAME, "integral: AbsTol must be nonnegative"));
        }
        if self.rel_tol < 0.0 {
            return Err(optim_error(NAME, "integral: RelTol must be nonnegative"));
        }
        if self.abs_tol == 0.0 && self.rel_tol == 0.0 {
            return Err(optim_error(
                NAME,
                "integral: AbsTol and RelTol cannot both be zero",
            ));
        }
        Ok(())
    }
}

fn option_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        other => Err(optim_error(
            NAME,
            format!("integral: option names must be strings, got {other:?}"),
        )),
    }
}

async fn scalar_bound(label: &str, value: Value) -> BuiltinResult<f64> {
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    let parsed = match value {
        Value::Num(n) => n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(b) => {
            if b {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            if data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(optim_error(
                NAME,
                format!("integral: {label} must be a finite real scalar, got {other:?}"),
            ))
        }
    };
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(optim_error(
            NAME,
            format!("integral: {label} must be finite"),
        ))
    }
}

fn numeric_option(name: &str, value: &Value) -> BuiltinResult<f64> {
    let parsed = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(b) => {
            if *b {
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
            return Err(optim_error(
                NAME,
                format!("integral: option {name} must be numeric, got {other:?}"),
            ))
        }
    };
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(optim_error(
            NAME,
            format!("integral: option {name} must be finite"),
        ))
    }
}

fn integer_option(name: &str, value: &Value) -> BuiltinResult<usize> {
    let parsed = numeric_option(name, value)?;
    if parsed < 0.0 {
        return Err(optim_error(
            NAME,
            format!("integral: option {name} must be nonnegative"),
        ));
    }
    Ok(parsed.floor() as usize)
}

fn bool_option(name: &str, value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Num(n) if *n == 0.0 || *n == 1.0 => Ok(*n != 0.0),
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw == 0 || raw == 1 {
                Ok(raw != 0)
            } else {
                Err(optim_error(
                    NAME,
                    format!("integral: option {name} must be logical scalar"),
                ))
            }
        }
        other => Err(optim_error(
            NAME,
            format!("integral: option {name} must be logical scalar, got {other:?}"),
        )),
    }
}

async fn integrate_finite_scalar(
    function: &Value,
    a: f64,
    b: f64,
    options: &IntegralOptions,
) -> BuiltinResult<f64> {
    let fa = call_integrand(function, a).await?;
    let m = 0.5 * (a + b);
    let fm = call_integrand(function, m).await?;
    let fb = call_integrand(function, b).await?;
    let mut evals = 3usize;
    let whole = simpson(a, b, fa, fm, fb);
    let tol = options.abs_tol.max(options.rel_tol * whole.abs());
    adaptive_simpson(
        function,
        SimpsonState {
            a,
            b,
            fa,
            fm,
            fb,
            whole,
            tol,
            depth: MAX_DEPTH,
        },
        &mut evals,
        options.max_fun_evals,
    )
    .await
}

#[derive(Clone, Copy)]
struct SimpsonState {
    a: f64,
    b: f64,
    fa: f64,
    fm: f64,
    fb: f64,
    whole: f64,
    tol: f64,
    depth: usize,
}

#[async_recursion::async_recursion(?Send)]
async fn adaptive_simpson(
    function: &Value,
    state: SimpsonState,
    evals: &mut usize,
    max_fun_evals: usize,
) -> BuiltinResult<f64> {
    if *evals + 2 > max_fun_evals {
        return Err(optim_error(
            NAME,
            "integral: exceeded maximum function evaluations",
        ));
    }

    let c = 0.5 * (state.a + state.b);
    let d = 0.5 * (state.a + c);
    let e = 0.5 * (c + state.b);
    let fd = call_integrand(function, d).await?;
    let fe = call_integrand(function, e).await?;
    *evals += 2;

    let left = simpson(state.a, c, state.fa, fd, state.fm);
    let right = simpson(c, state.b, state.fm, fe, state.fb);
    let refined = left + right;
    let error = refined - state.whole;
    if error.abs() <= 15.0 * state.tol {
        return Ok(refined + error / 15.0);
    }
    if state.depth == 0 {
        return Err(optim_error(
            NAME,
            "integral: adaptive quadrature did not converge",
        ));
    }

    let left_value = adaptive_simpson(
        function,
        SimpsonState {
            a: state.a,
            b: c,
            fa: state.fa,
            fm: fd,
            fb: state.fm,
            whole: left,
            tol: state.tol * 0.5,
            depth: state.depth - 1,
        },
        evals,
        max_fun_evals,
    )
    .await?;
    let right_value = adaptive_simpson(
        function,
        SimpsonState {
            a: c,
            b: state.b,
            fa: state.fm,
            fm: fe,
            fb: state.fb,
            whole: right,
            tol: state.tol * 0.5,
            depth: state.depth - 1,
        },
        evals,
        max_fun_evals,
    )
    .await?;
    Ok(left_value + right_value)
}

fn simpson(a: f64, b: f64, fa: f64, fm: f64, fb: f64) -> f64 {
    (b - a) * (fa + 4.0 * fm + fb) / 6.0
}

async fn call_integrand(function: &Value, x: f64) -> BuiltinResult<f64> {
    let value = call_function(function, vec![Value::Num(x)]).await?;
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    match value {
        Value::Num(n) if n.is_finite() => Ok(n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor.data.len() == 1 && tensor.data[0].is_finite() => {
            Ok(tensor.data[0])
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        Value::Num(_) | Value::Tensor(_) => Err(optim_error(
            NAME,
            "integral: function value must be a finite real scalar",
        )),
        other => Err(optim_error(
            NAME,
            format!("integral: function value must be real numeric scalar, got {other:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[runtime_builtin(
        name = "__integral_square",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn square_helper(x: Value) -> crate::BuiltinResult<Value> {
        let x = scalar_bound("x", x).await?;
        Ok(Value::Num(x * x))
    }

    #[runtime_builtin(
        name = "__integral_vector",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn vector_helper(_x: Value) -> crate::BuiltinResult<Value> {
        Ok(Value::Tensor(
            Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
        ))
    }

    #[runtime_builtin(
        name = "__integral_nan",
        type_resolver(crate::builtins::math::optim::type_resolvers::numerical_integral_type),
        builtin_path = "crate::builtins::math::optim::integral::tests"
    )]
    async fn nan_helper(_x: Value) -> crate::BuiltinResult<Value> {
        Ok(Value::Num(f64::NAN))
    }

    fn run(function: Value, a: f64, b: f64) -> crate::BuiltinResult<Value> {
        block_on(integral_builtin(
            function,
            Value::Num(a),
            Value::Num(b),
            Vec::new(),
        ))
    }

    #[test]
    fn integrates_named_sine_function() {
        let result = run(
            Value::FunctionHandle("sin".into()),
            0.0,
            std::f64::consts::PI,
        )
        .expect("integral");
        match result {
            Value::Num(value) => assert!((value - 2.0).abs() < 1.0e-7),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn integrates_polynomial_helper() {
        let result =
            run(Value::FunctionHandle("__integral_square".into()), 0.0, 1.0).expect("integral");
        match result {
            Value::Num(value) => assert!((value - (1.0 / 3.0)).abs() < 1.0e-9),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn reversed_bounds_negate_result() {
        let result = run(
            Value::FunctionHandle("sin".into()),
            std::f64::consts::PI,
            0.0,
        )
        .expect("integral");
        match result {
            Value::Num(value) => assert!((value + 2.0).abs() < 1.0e-7),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn zero_width_interval_returns_zero_without_callback() {
        let result =
            run(Value::FunctionHandle("__integral_nan".into()), 1.0, 1.0).expect("integral");
        assert!(matches!(result, Value::Num(0.0)));
    }

    #[test]
    fn rejects_vector_valued_integrand_for_initial_scope() {
        let err = run(Value::FunctionHandle("__integral_vector".into()), 0.0, 1.0).unwrap_err();
        assert!(err.message().contains("finite real scalar"));
    }

    #[test]
    fn rejects_nonfinite_integrand_values() {
        let err = run(Value::FunctionHandle("__integral_nan".into()), 0.0, 1.0).unwrap_err();
        assert!(err.message().contains("finite real scalar"));
    }

    #[test]
    fn accepts_tolerance_name_value_options() {
        let result = block_on(integral_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(0.0),
            Value::Num(std::f64::consts::PI),
            vec![
                Value::from("AbsTol"),
                Value::Num(1.0e-12),
                Value::from("RelTol"),
                Value::Num(1.0e-8),
            ],
        ))
        .expect("integral");
        match result {
            Value::Num(value) => assert!((value - 2.0).abs() < 1.0e-8),
            other => panic!("unexpected value {other:?}"),
        }
    }
}
