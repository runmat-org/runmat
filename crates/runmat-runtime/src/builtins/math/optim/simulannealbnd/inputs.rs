//! Complete positional/problem forms. Custom payloads are never traversed.
use super::{invalid, ScalarCooperation};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_value::{NumericDType, Tensor, Value};
pub(super) fn empty() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty"))
}
pub(super) fn is_empty(value: &Value) -> bool {
    matches!(value,Value::Tensor(v)if v.is_empty())
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
    ) || crate::builtins::common::validation::value_text_scalar_string(value)
        .is_some_and(|v| !v.trim().is_empty())
}
pub(super) fn doubles<'a>(value: &'a Value, role: &str) -> BuiltinResult<&'a [f64]> {
    match value {
        Value::Num(v) => Ok(std::slice::from_ref(v)),
        Value::Tensor(v) if v.numeric_dtype() == NumericDType::F64 => {
            Ok(v.as_f64_slice().expect("F64"))
        }
        _ => Err(invalid(format!(
            "{role} must be a concrete full real host double array"
        ))),
    }
}
pub(super) fn scalar(value: &Value, role: &str) -> BuiltinResult<f64> {
    let values = doubles(value, role)?;
    if values.len() != 1 {
        return Err(invalid(format!("{role} must be scalar")));
    }
    Ok(values[0])
}
pub(super) struct Problem {
    pub objective: RootedValue,
    pub initial: RootedValue,
    pub lower: Value,
    pub upper: Value,
    pub options: Value,
    pub rng: Option<crate::builtins::common::random::RngSnapshot>,
}
pub(super) async fn parse(
    args: crate::call::BuiltinArguments,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Problem> {
    let (mut values, rng) = if args.len() == 1 {
        let Value::Struct(problem) = &args[0] else {
            return Err(invalid("one argument requires a scalar problem structure"));
        };
        for name in ["objective", "x0", "lb", "ub", "solver", "options"] {
            if !problem.fields.contains_key(name) {
                return Err(invalid(format!("problem requires {name}")));
            }
        }
        if crate::builtins::common::validation::value_text_scalar_string(&problem.fields["solver"])
            .and_then(|v| runmat_builtins::OptimSolver::parse(&v))
            != Some(runmat_builtins::OptimSolver::Simulannealbnd)
        {
            return Err(invalid("problem solver must be simulannealbnd"));
        }
        let rng = problem
            .fields
            .get("rngstate")
            .map(|v| crate::builtins::stats::random::rng::restore_snapshot_value(v, c.runtime()))
            .transpose()?;
        (
            ["objective", "x0", "lb", "ub", "options"]
                .into_iter()
                .map(|n| problem.fields[n].clone())
                .collect::<Vec<_>>(),
            rng,
        )
    } else {
        if !matches!(args.len(), 2 | 4 | 5) {
            return Err(invalid(
                "expected fun,x0; fun,x0,lb,ub[,options]; or one complete problem",
            ));
        }
        let mut values = args.into_iter().collect::<Vec<_>>();
        values.resize_with(5, empty);
        (values, None)
    };
    if !callable(&values[0]) {
        return Err(invalid(
            "fun must be a function handle or scalar function name",
        ));
    }
    let options = values.pop().expect("five");
    let upper = values.pop().expect("five");
    let lower = values.pop().expect("five");
    let initial = RootedValue::in_optional_context(values.pop().expect("x0"), c.runtime())?;
    let target = {
        let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
        crate::canonicalize_callback_handle_for_semantic_resolution(values.pop().expect("fun"))?
    };
    let objective = RootedValue::in_optional_context(target, c.runtime())?;
    Ok(Problem {
        objective,
        initial,
        lower,
        upper,
        options,
        rng,
    })
}
pub(super) enum Domain {
    Double {
        shape: Vec<usize>,
        lower: Vec<f64>,
        upper: Vec<f64>,
    },
    /// Only outer extent metadata belongs here. Fields, captures and property
    /// values remain opaque to the objective/custom proposal.
    Custom { variables: usize },
}
pub(super) async fn preflight(problem: &Problem, c: &mut ScalarCooperation) -> BuiltinResult<()> {
    let datatype = if let Value::Struct(options) = &problem.options {
        options
            .fields
            .iter()
            .find_map(|(name, value)| {
                (runmat_builtins::OptimOptionField::parse_for_solver(
                    name,
                    runmat_builtins::OptimSolver::Simulannealbnd,
                ) == Some(runmat_builtins::OptimOptionField::DataType)
                    && !is_empty(value))
                .then_some(value)
            })
            .map(|value| {
                super::common::anneal_options::text(
                    super::common::anneal_options::Owner::Solver,
                    value,
                )
            })
            .transpose()?
            .unwrap_or_else(|| "double".to_owned())
    } else {
        "double".to_owned()
    };
    let datatype = runmat_builtins::OptimAnnealData::parse(&datatype)
        .ok_or_else(|| invalid("DataType must be double or custom"))?;
    if datatype == runmat_builtins::OptimAnnealData::Custom {
        return Ok(());
    }
    let x = doubles(problem.initial.value(), "x0")?;
    if x.is_empty() {
        return Err(invalid("x0 must be nonempty"));
    }
    if matches!(problem.initial.value(),Value::Tensor(v)if v.shape().len()>2||v.shape().iter().filter(|&&d|d!=1).count()>1)
    {
        return Err(invalid("x0 must be a row or column vector"));
    }
    for &v in x {
        if !v.is_finite() {
            return Err(invalid("x0 must be finite"));
        }
        c.completed(1).await?;
    }
    for (value, name) in [(&problem.lower, "lb"), (&problem.upper, "ub")] {
        let values = doubles(value, name)?;
        if values.len() > x.len() {
            return Err(invalid("bounds cannot have more entries than x0"));
        }
        for &v in values {
            if v.is_nan() {
                return Err(invalid("bounds cannot contain NaN"));
            }
            c.completed(1).await?;
        }
    }
    Ok(())
}
impl Domain {
    pub async fn admit(
        problem: &Problem,
        custom: bool,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<(Self, bool)> {
        if custom {
            if !is_empty(&problem.lower) || !is_empty(&problem.upper) {
                return Err(invalid(
                    "custom decision objects require empty numeric bounds",
                ));
            }
            let shape = match problem.initial.value() {
                Value::Tensor(v) => Some(v.shape()),
                Value::Cell(v) => Some(v.shape()),
                Value::StructArray(v) => Some(v.shape()),
                Value::ObjectArray(v) => Some(v.shape()),
                Value::ComplexTensor(v) => Some(v.shape()),
                Value::SymbolicArray(v) => Some(v.shape.as_slice()),
                Value::GpuTensor(v) => Some(v.shape()),
                Value::CharArray(v) => Some(v.shape()),
                Value::StringArray(v) => Some(v.shape()),
                Value::LogicalArray(v) => Some(v.shape.as_slice()),
                _ => None,
            };
            let variables = shape
                .map(|shape| shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)))
                .unwrap_or_else(|| match problem.initial.value() {
                    Value::SparseTensor(v) => v.rows.checked_mul(v.cols),
                    _ => Some(1),
                })
                .ok_or_else(|| invalid("custom outer extent overflow"))?;
            if variables == 0 {
                return Err(invalid("custom starting decision object cannot be empty"));
            }
            return Ok((Self::Custom { variables }, false));
        }
        let x = doubles(problem.initial.value(), "x0")?;
        if x.is_empty() {
            return Err(invalid("x0 must be a nonempty finite double vector"));
        }
        for &value in x {
            if !value.is_finite() {
                return Err(invalid("x0 must be finite"));
            }
            c.completed(1).await?;
        }
        let shape = match problem.initial.value() {
            Value::Tensor(v) => v.shape().to_vec(),
            _ => vec![1, 1],
        };
        if shape.len() > 2 || shape.iter().filter(|&&v| v != 1).count() > 1 {
            return Err(invalid("double x0 must be a row or column vector"));
        }
        let lower = bound(&problem.lower, x.len(), f64::NEG_INFINITY, "lb", c).await?;
        let upper = bound(&problem.upper, x.len(), f64::INFINITY, "ub", c).await?;
        let mut impossible = false;
        for (&a, &b) in lower.iter().zip(&upper) {
            impossible |= a > b || a == f64::INFINITY || b == f64::NEG_INFINITY;
            c.completed(1).await?;
        }
        Ok((
            Self::Double {
                shape,
                lower,
                upper,
            },
            impossible,
        ))
    }
    pub fn variables(&self) -> usize {
        match self {
            Self::Double { lower, .. } => lower.len(),
            Self::Custom { variables } => *variables,
        }
    }
    pub fn bounded(&self) -> bool {
        match self {
            Self::Double { lower, upper, .. } => lower
                .iter()
                .zip(upper)
                .any(|(&a, &b)| a.is_finite() || b.is_finite()),
            Self::Custom { .. } => false,
        }
    }
    pub fn value(&self, values: Vec<f64>) -> BuiltinResult<Value> {
        let Self::Double { shape, .. } = self else {
            return Err(invalid("numeric publication requires double domain"));
        };
        Tensor::new(values, shape.clone())
            .map(Value::Tensor)
            .map_err(invalid)
    }
    pub async fn initial(
        &self,
        value: &Value,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<RootedValue> {
        let value = match self {
            Self::Custom { .. } => value.clone(),
            Self::Double { lower, upper, .. } => {
                let mut outside = false;
                for ((&v, &a), &b) in doubles(value, "x0")?.iter().zip(lower).zip(upper) {
                    outside |= v < a || v > b;
                    c.completed(1).await?;
                }
                if !outside {
                    return RootedValue::in_optional_context(value.clone(), c.runtime());
                }
                let mut values = Vec::with_capacity(lower.len());
                for ((&v, &a), &b) in doubles(value, "x0")?.iter().zip(lower).zip(upper) {
                    values.push(v.max(a).min(b));
                    c.completed(1).await?;
                }
                self.value(values)?
            }
        };
        RootedValue::in_optional_context(value, c.runtime())
    }
    pub async fn validate_trial(
        &self,
        value: Value,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<RootedValue> {
        if let Self::Double {
            shape,
            lower,
            upper,
        } = self
        {
            let x = doubles(&value, "proposal")?;
            if x.len() != lower.len() {
                return Err(invalid("proposal must contain exactly numel(x0) doubles"));
            }
            for ((&v, &a), &b) in x.iter().zip(lower).zip(upper) {
                if !v.is_finite() || v < a || v > b {
                    return Err(invalid("custom proposal must be finite and honor bounds"));
                }
                c.completed(1).await?;
            }
            let value = match value {
                Value::Tensor(tensor) => {
                    Value::Tensor(tensor.reshape(shape.clone()).map_err(invalid)?)
                }
                Value::Num(value) => self.value(vec![value])?,
                _ => unreachable!("validated double carrier"),
            };
            return RootedValue::in_optional_context(value, c.runtime());
        }
        RootedValue::in_optional_context(value, c.runtime())
    }
}
async fn bound(
    value: &Value,
    n: usize,
    default: f64,
    name: &str,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let source = doubles(value, name)?;
    if source.len() > n || source.iter().any(|v| v.is_nan()) {
        return Err(invalid(format!("{name} has an invalid extent or NaN")));
    }
    if !source.is_empty() && source.len() < n {
        let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
        crate::console::record_console_line(crate::console::ConsoleStream::Stderr,
            format!("Warning: simulannealbnd: {name} has fewer entries than x0; remaining coordinates are unbounded."))?;
    }
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        values.push(source.get(i).copied().unwrap_or(default));
        c.completed(1).await?;
    }
    Ok(values)
}
