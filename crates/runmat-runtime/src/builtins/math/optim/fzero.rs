//! MATLAB-compatible `fzero` builtin for scalar nonlinear root finding.

use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::optim::common::{
    call_scalar_function, optim_error, option_f64, option_string, option_usize,
};
use crate::builtins::math::optim::type_resolvers::scalar_root_type;
use crate::BuiltinResult;

const NAME: &str = "fzero";
const DEFAULT_TOL_X: f64 = 1.0e-6;
const DEFAULT_MAX_ITER: usize = 400;
const DEFAULT_MAX_FUN_EVALS: usize = 500;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::fzero")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fzero",
    op_kind: GpuOpKind::Custom("scalar-root-find"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host iterative solver. Callback values may use GPU-aware builtins, but the root search runs on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::fzero")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fzero",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Root finding repeatedly invokes user code and terminates fusion planning.",
};

#[runtime_builtin(
    name = "fzero",
    category = "math/optim",
    summary = "Find a zero of a scalar nonlinear function using bracket expansion and Brent's method.",
    keywords = "fzero,root finding,zero,brent,optimization",
    accel = "sink",
    type_resolver(scalar_root_type),
    builtin_path = "crate::builtins::math::optim::fzero"
)]
async fn fzero_builtin(function: Value, x: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(optim_error(NAME, "fzero: too many input arguments"));
    }
    let options = parse_options(rest.first())?;
    let opts = FzeroOptions::from_struct(options.as_ref())?;
    let bracket = initial_bracket(&function, x, &opts).await?;
    let root = brent(&function, bracket, &opts).await?;
    Ok(Value::Num(root))
}

fn parse_options(value: Option<&Value>) -> BuiltinResult<Option<StructValue>> {
    match value {
        None => Ok(None),
        Some(Value::Struct(options)) => Ok(Some(options.clone())),
        Some(other) => Err(optim_error(
            NAME,
            format!("fzero: options must be a struct, got {other:?}"),
        )),
    }
}

#[derive(Clone, Copy)]
struct FzeroOptions {
    tol_x: f64,
    max_iter: usize,
    max_fun_evals: usize,
}

impl FzeroOptions {
    fn from_struct(options: Option<&StructValue>) -> BuiltinResult<Self> {
        let display = option_string(options, "Display", "off")?;
        if !matches!(display.as_str(), "off" | "none" | "final" | "iter") {
            return Err(optim_error(
                NAME,
                "fzero: option Display must be 'off', 'none', 'final', or 'iter'",
            ));
        }
        let tol_x = option_f64(NAME, options, "TolX", DEFAULT_TOL_X)?;
        if tol_x <= 0.0 {
            return Err(optim_error(NAME, "fzero: option TolX must be positive"));
        }
        let max_iter = option_usize(NAME, options, "MaxIter", DEFAULT_MAX_ITER)?;
        let max_fun_evals = option_usize(NAME, options, "MaxFunEvals", DEFAULT_MAX_FUN_EVALS)?;
        Ok(Self {
            tol_x,
            max_iter: max_iter.max(1),
            max_fun_evals: max_fun_evals.max(1),
        })
    }
}

#[derive(Clone, Copy)]
struct Bracket {
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    evals: usize,
}

async fn initial_bracket(
    function: &Value,
    x: Value,
    options: &FzeroOptions,
) -> BuiltinResult<Bracket> {
    let x = crate::dispatcher::gather_if_needed_async(&x).await?;
    match x {
        Value::Tensor(tensor) if tensor.data.len() == 2 => {
            let a = tensor.data[0];
            let b = tensor.data[1];
            bracket_from_endpoints(function, a, b).await
        }
        Value::Tensor(tensor) if tensor.data.len() == 1 => {
            expand_bracket(function, tensor.data[0], options).await
        }
        Value::Num(n) => expand_bracket(function, n, options).await,
        Value::Int(i) => expand_bracket(function, i.to_f64(), options).await,
        Value::Bool(b) => expand_bracket(function, if b { 1.0 } else { 0.0 }, options).await,
        other => Err(optim_error(
            NAME,
            format!("fzero: initial point must be a scalar or two-element bracket, got {other:?}"),
        )),
    }
}

async fn bracket_from_endpoints(function: &Value, a: f64, b: f64) -> BuiltinResult<Bracket> {
    if !a.is_finite() || !b.is_finite() || a == b {
        return Err(optim_error(
            NAME,
            "fzero: bracket endpoints must be finite and distinct",
        ));
    }
    let fa = call_scalar_function(NAME, function, a).await?;
    if fa == 0.0 {
        return Ok(Bracket {
            a,
            b: a,
            fa,
            fb: fa,
            evals: 1,
        });
    }
    let fb = call_scalar_function(NAME, function, b).await?;
    if fb == 0.0 || fa.signum() != fb.signum() {
        Ok(Bracket {
            a,
            b,
            fa,
            fb,
            evals: 2,
        })
    } else {
        Err(optim_error(
            NAME,
            "fzero: function values at bracket endpoints must differ in sign",
        ))
    }
}

async fn expand_bracket(
    function: &Value,
    x0: f64,
    options: &FzeroOptions,
) -> BuiltinResult<Bracket> {
    if !x0.is_finite() {
        return Err(optim_error(NAME, "fzero: initial point must be finite"));
    }
    let f0 = call_scalar_function(NAME, function, x0).await?;
    if f0 == 0.0 {
        return Ok(Bracket {
            a: x0,
            b: x0,
            fa: f0,
            fb: f0,
            evals: 1,
        });
    }

    let mut evals = 1usize;
    let mut step = (x0.abs() * 0.01).max(0.01);
    while evals + 2 <= options.max_fun_evals {
        let a = x0 - step;
        let b = x0 + step;
        let fa = call_scalar_function(NAME, function, a).await?;
        let fb = call_scalar_function(NAME, function, b).await?;
        evals += 2;
        if fa == 0.0 {
            return Ok(Bracket {
                a,
                b: a,
                fa,
                fb: fa,
                evals,
            });
        }
        if fa.signum() != f0.signum() {
            return Ok(Bracket {
                a,
                b: x0,
                fa,
                fb: f0,
                evals,
            });
        }
        if fb.signum() != f0.signum() {
            return Ok(Bracket {
                a: x0,
                b,
                fa: f0,
                fb,
                evals,
            });
        }
        if fb == 0.0 || fa.signum() != fb.signum() {
            return Ok(Bracket {
                a,
                b,
                fa,
                fb,
                evals,
            });
        }
        step *= 1.6;
    }

    Err(optim_error(
        NAME,
        "fzero: could not find a sign-changing bracket around the initial point",
    ))
}

async fn brent(function: &Value, bracket: Bracket, options: &FzeroOptions) -> BuiltinResult<f64> {
    if bracket.fa == 0.0 || bracket.a == bracket.b {
        return Ok(bracket.a);
    }
    if bracket.fb == 0.0 {
        return Ok(bracket.b);
    }

    let mut a = bracket.a;
    let mut b = bracket.b;
    let mut c = a;
    let mut fa = bracket.fa;
    let mut fb = bracket.fb;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;
    let mut evals = bracket.evals;

    for _ in 0..options.max_iter {
        if fb.signum() == fc.signum() {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            let old_b = b;
            let old_fb = fb;
            a = b;
            fa = fb;
            b = c;
            fb = fc;
            c = old_b;
            fc = old_fb;
        }

        let tol = 2.0 * f64::EPSILON * b.abs() + 0.5 * options.tol_x;
        let midpoint = 0.5 * (c - b);
        if midpoint.abs() <= tol || fb == 0.0 {
            return Ok(b);
        }
        if evals >= options.max_fun_evals {
            return Err(optim_error(
                NAME,
                "fzero: exceeded maximum function evaluations",
            ));
        }

        if e.abs() >= tol && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut p, mut q) = if a == c {
                (2.0 * midpoint * s, 1.0 - s)
            } else {
                let q = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * midpoint * q * (q - r) - (b - a) * (r - 1.0)),
                    (q - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            if p > 0.0 {
                q = -q;
            }
            p = p.abs();
            if interpolation_step_accepted(p, q, midpoint, tol, e) {
                e = d;
                d = p / q;
            } else {
                d = midpoint;
                e = d;
            }
        } else {
            d = midpoint;
            e = d;
        }

        a = b;
        fa = fb;
        b += if d.abs() > tol {
            d
        } else if midpoint >= 0.0 {
            tol
        } else {
            -tol
        };
        fb = call_scalar_function(NAME, function, b).await?;
        evals += 1;
    }

    Err(optim_error(NAME, "fzero: exceeded maximum iterations"))
}

fn interpolation_step_accepted(p: f64, q: f64, midpoint: f64, tol: f64, e: f64) -> bool {
    let min_a = 3.0 * midpoint * q - (tol * q).abs();
    let min_b = (e * q).abs();
    2.0 * p < min_a.min(min_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    #[test]
    fn fzero_bracketed_builtin_handle() {
        let bracket = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let root = block_on(fzero_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Tensor(bracket),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Num(n) => assert!((n - std::f64::consts::PI).abs() < 1.0e-6),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fzero_scalar_initial_guess_expands_bracket() {
        let root = block_on(fzero_builtin(
            Value::FunctionHandle("cos".into()),
            Value::Num(1.0),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Num(n) => assert!((n - std::f64::consts::FRAC_PI_2).abs() < 1.0e-6),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn fzero_scalar_initial_guess_uses_center_sign_for_bracket() {
        let root = block_on(fzero_builtin(
            Value::FunctionHandle("sin".into()),
            Value::Num(std::f64::consts::FRAC_PI_2),
            Vec::new(),
        ))
        .unwrap();
        match root {
            Value::Num(n) => assert!(n.abs() < 1.0e-6),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn brent_interpolation_acceptance_uses_signed_q() {
        assert!(!interpolation_step_accepted(1.0, -2.0, 1.0, 0.1, 10.0));
        assert!(interpolation_step_accepted(1.0, -2.0, -1.0, 0.1, 10.0));
    }
}
