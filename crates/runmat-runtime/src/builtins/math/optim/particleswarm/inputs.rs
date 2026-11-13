//! Positional/problem grammar and concrete documented double carriers.
use super::{invalid, population::capacity, ScalarCooperation};
use crate::BuiltinResult;
use runmat_value::{NumericDType, Tensor, Value};
pub(super) struct Problem {
    pub objective: Value,
    pub variables: usize,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub options: Value,
    pub rng: Option<crate::builtins::common::random::RngSnapshot>,
    pub inconsistent: bool,
}
pub(super) fn empty() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty shape"))
}
pub(super) fn callable(value: &Value) -> bool {
    matches!(
        value,
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
            | Value::String(_)
            | Value::CharArray(_)
    ) || matches!(value,Value::StringArray(v)if v.data().len()==1)
}
pub(super) fn double(value: &Value, role: &str) -> BuiltinResult<()> {
    if matches!(value, Value::Num(_))
        || matches!(value,Value::Tensor(v)if v.numeric_dtype()==NumericDType::F64)
    {
        Ok(())
    } else {
        Err(invalid(format!(
            "{role} must be full real host double data"
        )))
    }
}
pub(super) async fn parse(
    args: crate::call::BuiltinArguments,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Problem> {
    parse_role(args, ParseRole::Solver, cooperation).await
}
pub(super) async fn parse_creation(
    args: crate::call::BuiltinArguments,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Problem> {
    parse_role(args, ParseRole::Creation, cooperation).await
}
enum ParseRole {
    Solver,
    Creation,
}
async fn parse_role(
    args: crate::call::BuiltinArguments,
    role: ParseRole,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Problem> {
    let (mut values, rng) = if args.len() == 1 {
        let Value::Struct(problem) = args.into_iter().next().unwrap() else {
            return Err(invalid(
                "single argument form requires a scalar problem structure",
            ));
        };
        for name in ["objective", "nvars", "lb", "ub", "solver", "options"] {
            if !problem.fields.contains_key(name) {
                return Err(invalid(format!("problem requires {name}")));
            }
        }
        if crate::builtins::common::validation::value_text_scalar_string(&problem.fields["solver"])
            .and_then(|name| runmat_builtins::OptimSolver::parse(&name))
            != Some(runmat_builtins::OptimSolver::Particleswarm)
        {
            return Err(invalid("problem solver must be particleswarm"));
        }
        let rng = if matches!(role, ParseRole::Solver) {
            problem
                .fields
                .get("rngstate")
                .map(|value| {
                    crate::builtins::stats::random::rng::restore_snapshot_value(
                        value,
                        cooperation.runtime(),
                    )
                })
                .transpose()?
        } else {
            None
        };
        (
            ["objective", "nvars", "lb", "ub", "options"]
                .into_iter()
                .map(|name| problem.fields.get(name).cloned().unwrap_or_else(empty))
                .collect::<Vec<_>>(),
            rng,
        )
    } else {
        if !matches!(args.len(), 2 | 4 | 5) {
            return Err(invalid(
                "expected fun,nvars, an optional complete lb/ub pair, then options",
            ));
        }
        let mut values = args.into_iter().collect::<Vec<_>>();
        values.resize_with(5, empty);
        (values, None)
    };
    if !callable(&values[0]) {
        return Err(invalid(
            "fun must be a single function handle or function name",
        ));
    }
    for (index, name) in [(1, "nvars"), (2, "lb"), (3, "ub")] {
        double(&values[index], name)?;
    }
    let variables = super::common::swarm_options::own_parse(
        super::common::swarm_options::Owner::Solver,
        crate::builtins::math::optim::common::option_usize_value(
            "particleswarm",
            Some(&values[1]),
            "nvars",
            0,
        ),
    )?;
    if variables == 0 {
        return Err(invalid(
            "nvars must be a positive integer-valued double scalar",
        ));
    }
    let options = values.pop().unwrap();
    let upper = bound(
        values.pop().unwrap(),
        variables,
        f64::INFINITY,
        "ub",
        cooperation,
    )
    .await?;
    let lower = bound(
        values.pop().unwrap(),
        variables,
        f64::NEG_INFINITY,
        "lb",
        cooperation,
    )
    .await?;
    let mut inconsistent = false;
    for (&lower, &upper) in lower.iter().zip(&upper) {
        inconsistent |= lower > upper || lower == f64::INFINITY || upper == f64::NEG_INFINITY;
        cooperation.completed(1).await?;
    }
    let objective = values.remove(0);
    Ok(Problem {
        objective,
        variables,
        lower,
        upper,
        options,
        rng,
        inconsistent,
    })
}
async fn bound(
    value: Value,
    n: usize,
    default: f64,
    name: &str,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let source = match &value {
        Value::Num(v) => std::slice::from_ref(v),
        Value::Tensor(v) => v.as_f64_slice().expect("concrete double bounds"),
        _ => unreachable!("admitted bound carrier"),
    };
    if source.len() > n {
        return Err(invalid(format!("{name} has more than nvars entries")));
    }
    for &v in source {
        if v.is_nan() {
            return Err(invalid(format!("{name} cannot contain NaN")));
        }
        cooperation.completed(1).await?;
    }
    if source.len() > 1 && source.len() < n {
        let _scope = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        crate::console::record_console_line(crate::console::ConsoleStream::Stderr,
            format!("Warning: particleswarm: {name} has fewer entries than nvars; remaining coordinates are unbounded."))?;
    }
    let mut values = capacity(n, cooperation)?;
    for index in 0..n {
        let value = if source.len() == 1 {
            source[0]
        } else {
            source.get(index).copied().unwrap_or(default)
        };
        values.push(value);
        cooperation.completed(1).await?;
    }
    Ok(values)
}
pub(super) async fn problem_value(
    problem: &Problem,
    options: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValue> {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("solver", Value::from("particleswarm"));
    fields.insert("objective", problem.objective.clone());
    fields.insert("nvars", Value::Num(problem.variables as f64));
    for (name, values) in [("lb", &problem.lower), ("ub", &problem.upper)] {
        let mut out = capacity(values.len(), cooperation)?;
        for &v in values {
            out.push(v);
            cooperation.completed(1).await?;
        }
        fields.insert(
            name,
            Value::Tensor(Tensor::new(out, vec![1, values.len()]).map_err(invalid)?),
        );
    }
    fields.insert("options", options);
    crate::execution::RootedValue::in_optional_context(
        Value::Struct(fields.finish()),
        cooperation.runtime(),
    )
}
