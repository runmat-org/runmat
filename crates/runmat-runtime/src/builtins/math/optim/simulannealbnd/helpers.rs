//! Public predefined operators and real typed progress plots.
#[cfg(feature = "plot-core")]
use super::common;
use super::{inputs, invalid, operators, random, ScalarCooperation};
use crate::BuiltinResult;
use runmat_builtins::{OptimPlotKind as Plot, OptimTemperature};
use runmat_value::Value;
#[derive(Clone, Copy)]
enum Role {
    Acceptance,
    FastProposal,
    BoltzProposal,
    Exponential,
    FastTemperature,
    BoltzTemperature,
    BestFunction,
    BestPoint,
    CurrentFunction,
    CurrentPoint,
    Stopping,
    Temperature,
}
impl Role {
    fn name(self) -> &'static str {
        match self {
            Self::Acceptance => "acceptancesa",
            Self::FastProposal => "annealingfast",
            Self::BoltzProposal => "annealingboltz",
            Self::Exponential => "temperatureexp",
            Self::FastTemperature => "temperaturefast",
            Self::BoltzTemperature => "temperatureboltz",
            Self::BestFunction => "saplotbestf",
            Self::BestPoint => "saplotbestx",
            Self::CurrentFunction => "saplotf",
            Self::CurrentPoint => "saplotx",
            Self::Stopping => "saplotstopping",
            Self::Temperature => "saplottemperature",
        }
    }
    fn plot(self) -> Option<Plot> {
        Some(match self {
            Self::BestFunction => Plot::AnnealBestFunction,
            Self::BestPoint => Plot::AnnealBestPoint,
            Self::CurrentFunction => Plot::AnnealCurrentFunction,
            Self::CurrentPoint => Plot::AnnealCurrentPoint,
            Self::Stopping => Plot::AnnealStopping,
            Self::Temperature => Plot::AnnealTemperature,
            _ => return None,
        })
    }
}
fn field<'a>(value: &'a Value, name: &str) -> BuiltinResult<&'a Value> {
    let Value::Struct(fields) = value else {
        return Err(invalid("operator metadata must be a scalar structure"));
    };
    fields
        .fields
        .get(name)
        .ok_or_else(|| invalid(format!("operator metadata requires {name}")))
}
fn scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    inputs::scalar(field(value, name)?, name)
}
async fn run(
    role: Role,
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::context::host_allocation::require_unbounded_workspace(runtime,role.name(),
        "finite host-memory execution is unsupported; operator and scene workspaces lack a complete finite bound")?;
    if crate::output_count::current_output_count_in(runtime).is_some_and(|n| n > 1) {
        return Err(invalid("annealing helpers return at most one output"));
    }
    let expected = if role.plot().is_some() || matches!(role, Role::Acceptance) {
        3
    } else {
        2
    };
    if args.len() != expected {
        return Err(invalid(format!(
            "{} requires {expected} arguments",
            role.name()
        )));
    }
    let _original = args.fork();
    let mut c = ScalarCooperation::for_builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: role.name() },
        runtime,
    );
    c.completed(0).await?;
    let value = if let Some(plot) = role.plot() {
        draw(plot, &args, &mut c).await?
    } else {
        match role {
            Role::Acceptance => {
                let old = scalar(&args[0], "fval")?;
                let new = inputs::scalar(&args[2], "new fval")?;
                let temperatures = inputs::doubles(field(&args[0], "temperature")?, "temperature")?;
                if !old.is_finite()
                    || !new.is_finite()
                    || temperatures.is_empty()
                    || temperatures.iter().any(|v| !v.is_finite() || *v < 0.0)
                {
                    return Err(invalid(
                        "acceptance requires finite objective values and nonnegative temperature",
                    ));
                }
                let t = temperatures.iter().copied().fold(0.0, f64::max);
                Value::Bool(
                    new < old
                        || random::uniform(1, &mut c).await?[0]
                            < operators::probability(old, new, t),
                )
            }
            Role::FastProposal | Role::BoltzProposal => {
                let x = field(&args[0], "x")?;
                let values = inputs::doubles(x, "current x")?;
                let t = inputs::doubles(field(&args[0], "temperature")?, "temperature")?;
                if t.len() != values.len()
                    || values.is_empty()
                    || t.iter().any(|v| !v.is_finite() || *v < 0.0)
                {
                    return Err(invalid("proposal temperature extent/range mismatch"));
                }
                let direction = random::direction(values.len(), &mut c).await?;
                let mut trial = Vec::with_capacity(values.len());
                for i in 0..values.len() {
                    let scale = if matches!(role, Role::FastProposal) {
                        t[i]
                    } else {
                        t[i].sqrt()
                    };
                    trial.push(values[i] + scale * direction[i]);
                    c.completed(1).await?;
                }
                let lower =
                    expanded_bound(field(&args[1], "lb")?, values.len(), f64::NEG_INFINITY)?;
                let upper = expanded_bound(field(&args[1], "ub")?, values.len(), f64::INFINITY)?;
                let trial = operators::honor_bounds(trial, values, &lower, &upper, &mut c).await?;
                let shape = match x {
                    Value::Tensor(v) => v.shape().to_vec(),
                    _ => vec![1, 1],
                };
                Value::Tensor(runmat_value::Tensor::new(trial, shape).map_err(invalid)?)
            }
            Role::Exponential | Role::FastTemperature | Role::BoltzTemperature => {
                let k = inputs::doubles(field(&args[0], "k")?, "k")?;
                let initial =
                    inputs::doubles(field(&args[1], "InitialTemperature")?, "InitialTemperature")?;
                if k.is_empty() || initial.len() != 1 && initial.len() != k.len() {
                    return Err(invalid("temperature parameter extent mismatch"));
                }
                let kind = match role {
                    Role::Exponential => OptimTemperature::Exponential,
                    Role::FastTemperature => OptimTemperature::Fast,
                    _ => OptimTemperature::Boltzmann,
                };
                let mut values = Vec::with_capacity(k.len());
                for (i, &k) in k.iter().enumerate() {
                    if !k.is_finite() || k < 0.0 || (kind == OptimTemperature::Fast && k == 0.0) {
                        return Err(invalid(
                            "k must be finite and nonnegative, and positive for temperaturefast",
                        ));
                    }
                    values.push(operators::temperature(
                        kind,
                        initial[if initial.len() == 1 { 0 } else { i }],
                        k,
                    )?);
                    c.completed(1).await?;
                }
                Value::Tensor(runmat_value::Tensor::new(values, vec![k.len(), 1]).map_err(invalid)?)
            }
            _ => unreachable!("plot handled"),
        }
    };
    if crate::output_count::current_output_count_in(runtime) == Some(0) {
        Ok(Value::output_list(Vec::new()))
    } else {
        Ok(value)
    }
}
fn expanded_bound(value: &Value, n: usize, default: f64) -> BuiltinResult<Vec<f64>> {
    let values = inputs::doubles(value, "bounds")?;
    if values.len() > n || values.iter().any(|v| v.is_nan()) {
        return Err(invalid("invalid bound extent or NaN"));
    }
    Ok((0..n)
        .map(|i| values.get(i).copied().unwrap_or(default))
        .collect())
}
async fn draw(
    kind: Plot,
    args: &crate::call::BuiltinArguments,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = (kind, args, c);
        return Err(invalid("annealing plots require plot-core/UI"));
    }
    #[cfg(feature = "plot-core")]
    {
        let phase = crate::builtins::common::validation::value_text_scalar_string(&args[2])
            .and_then(|s| common::output_callback::OutputState::parse(&s))
            .ok_or_else(|| invalid("plot state must be init,iter,done"))?;
        let iteration = scalar(&args[1], "iteration")?;
        if !iteration.is_finite() || iteration < 0.0 || iteration.fract() != 0.0 {
            return Err(invalid("iteration must be finite nonnegative integral"));
        }
        let mut point = Vec::new();
        let value = match kind {
            Plot::AnnealBestPoint | Plot::AnnealCurrentPoint => {
                let name = if kind == Plot::AnnealBestPoint {
                    "bestx"
                } else {
                    "x"
                };
                for &v in inputs::doubles(field(&args[1], name)?, name)? {
                    if !v.is_finite() {
                        return Err(invalid("point plot requires finite coordinates"));
                    }
                    point.push(v);
                    c.completed(1).await?;
                }
                0.0
            }
            Plot::AnnealBestFunction => scalar(&args[1], "bestfval")?,
            Plot::AnnealCurrentFunction => scalar(&args[1], "fval")?,
            Plot::AnnealTemperature => {
                let values = inputs::doubles(field(&args[1], "temperature")?, "temperature")?;
                if values.is_empty() {
                    return Err(invalid("temperature must be nonempty"));
                }
                let mut mean = 0.0;
                for &v in values {
                    if !v.is_finite() || v < 0.0 {
                        return Err(invalid("temperature must be finite nonnegative"));
                    }
                    mean += v / values.len() as f64;
                    c.completed(1).await?;
                }
                mean
            }
            Plot::AnnealStopping => {
                let owner = common::anneal_options::Owner::Solver;
                let tolerance =
                    common::anneal_options::number(owner, field(&args[0], "FunctionTolerance")?)?;
                let change = scalar(&args[1], "averagechange")?;
                let fun = scalar(&args[1], "funccount")?;
                let count_limit = |value: &Value| -> BuiltinResult<f64> {
                    if common::anneal_options::infinite(value) {
                        return Ok(f64::INFINITY);
                    }
                    common::option_usize_value("simulannealbnd", Some(value), "plot count limit", 0)
                        .map(|v| v as f64)
                };
                let maximum = count_limit(field(&args[0], "MaxFunctionEvaluations")?)?;
                let iterations = count_limit(field(&args[0], "MaxIterations")?)?;
                // Solver metadata's t0 is from this original admitted clock.
                let started = scalar(&args[1], "t0")?;
                let runtime = c
                    .runtime()
                    .ok_or_else(common::solver_timing::SolverTiming::unavailable)?;
                let now = runtime.service_ports().timing()?.monotonic_elapsed();
                let started = std::time::Duration::try_from_secs_f64(started).map_err(|_| {
                    invalid("plot t0 is not a representable nonnegative start time")
                })?;
                let elapsed = now.saturating_sub(started);
                let limit = common::anneal_options::time(owner, field(&args[0], "MaxTime")?)?;
                let fraction = |v: f64, limit: f64| {
                    if limit == 0.0 {
                        1.0
                    } else {
                        (v / limit).clamp(0.0, 1.0)
                    }
                };
                point = vec![
                    if tolerance == 0.0
                        || !matches!(&args[1],Value::Struct(v)if v.fields.get("stallwindowfull").is_some_and(|v|matches!(v,Value::Bool(true))))
                    {
                        0.0
                    } else if change == 0.0 {
                        1.0
                    } else {
                        fraction(tolerance, change)
                    },
                    fraction(fun, maximum),
                    fraction(iteration, iterations),
                    limit.fraction(elapsed).unwrap_or(0.0).clamp(0.0, 1.0),
                ];
                0.0
            }
            _ => return Err(invalid("not an annealing plot role")),
        };
        if !value.is_finite() || point.iter().any(|v| !v.is_finite()) {
            return Err(invalid("plot metric is nonfinite"));
        }
        crate::builtins::plotting::state::update_optimization_plot(
            kind,
            phase,
            point,
            iteration,
            value,
            c.runtime(),
        )?;
        Ok(Value::Bool(false))
    }
}

pub(crate) mod bindings;
