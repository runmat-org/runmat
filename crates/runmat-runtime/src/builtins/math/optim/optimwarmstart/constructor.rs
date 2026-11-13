//! Host grammar before the one immutable native publication.
use super::*;
use crate::builtins::math::optim::common::{self, constrained_numeric as numeric};
pub(super) async fn create(
    args: crate::call::FunctionArgumentView<'_>,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if !matches!(args.len(), 2 | 6) {
        return Err(policy().invalid(
            "expected x0, active lsqlin or quadprog options, and optionally both memory-limit pairs",
        ));
    }
    let _roots = policy().roots(args.iter(), c.runtime())?;
    let input = numeric::Policy {
        owner: OPTIMWARMSTART_CATALOG_ENTRY.identity,
        invalid: Some(&OPTIMWARMSTART_ERROR_INVALID_INPUT),
        numeric: None,
        single: None,
        resident: None,
        sparse: None,
    };
    if !numeric::preflight(
        &args[0],
        "x0",
        numeric::Geometry::Array,
        &input,
        c.runtime(),
    )? {
        return Err(policy().invalid("x0 must be a real finite double array"));
    }
    let Value::Struct(options) = &args[1] else {
        return Err(policy().invalid("options must identify active-set lsqlin or quadprog"));
    };
    let solver = options
        .fields
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("Solver"))
        .map(|(_, v)| v)
        .ok_or_else(|| policy().invalid("options require a supported Solver profile"))?;
    let kind = match runmat_builtins::OptimSolver::parse(&common::field_name(solver)?) {
        Some(runmat_builtins::OptimSolver::Lsqlin) => Kind::LeastSquares,
        Some(runmat_builtins::OptimSolver::Quadprog) => Kind::Quadratic,
        _ => return Err(policy().invalid("warm starts require lsqlin or quadprog")),
    };
    let mut equalities = None;
    let mut inequalities = None;
    for pair in (2..args.len()).step_by(2) {
        let name = policy().text(&args[pair], c.runtime())?;
        let property = Property::parse_argument(name.as_str())
            .ok_or_else(|| policy().invalid("unknown warm-start memory-limit property"))?;
        let value = limit(&args[pair + 1], property.name())?;
        let previous = match property {
            Property::MaxLinearEqualities => equalities.replace(value),
            Property::MaxLinearInequalities => inequalities.replace(value),
            _ => return Err(policy().invalid("warm-start constructor properties are read-only")),
        };
        if previous.is_some() {
            return Err(policy().invalid("duplicate warm-start memory limit"));
        }
        c.completed(1).await?;
    }
    if equalities.is_some() != inequalities.is_some() {
        return Err(policy().invalid("supply both MaxLinearEqualities and MaxLinearInequalities"));
    }
    match kind {
        Kind::LeastSquares => super::super::lsqlin::validate_warm_options(&args[1], c).await?,
        Kind::Quadratic => super::super::quadprog::validate_warm_options(&args[1], c).await?,
    }
    let Some(x) =
        numeric::prepare(args[0].clone(), "x0", numeric::Geometry::Array, &input, c).await?
    else {
        return Err(policy().invalid("x0 must be a real finite double array"));
    };
    let x = match x {
        Value::Num(x) if x.is_finite() => {
            Tensor::new(vec![x], vec![1, 1]).map_err(|e| policy().invalid(e))?
        }
        Value::Tensor(value) => {
            let values = value
                .as_f64_slice()
                .ok_or_else(|| policy().invalid("x0 must retain double storage"))?;
            for chunk in values.chunks(1024) {
                if chunk.iter().any(|v| !v.is_finite()) {
                    return Err(policy().invalid("x0 must be finite"));
                }
                c.completed(chunk.len() as u64).await?;
            }
            let n = value.len();
            value.reshape(vec![n, 1]).map_err(|e| policy().invalid(e))?
        }
        _ => return Err(policy().invalid("x0 must be a real finite double array")),
    };
    let limits = WarmStartMemoryLimits {
        equalities: equalities.unwrap_or(None),
        inequalities: inequalities.unwrap_or(None),
    };
    let value = publish_value(kind, x, options.clone(), limits, None, c.runtime())?;
    RootedValue::in_optional_context(value, c.runtime())
}
fn limit(value: &Value, name: &str) -> BuiltinResult<Option<usize>> {
    let number = match value {
        Value::Num(number) => *number,
        Value::Tensor(value)
            if value.len() == 1
                && matches!(
                    value.numeric_dtype(),
                    runmat_value::NumericDType::F32 | runmat_value::NumericDType::F64
                ) =>
        {
            crate::builtins::common::tensor::tensor_value_f64(value, 0)
        }
        _ => {
            return Err(policy().invalid("memory limits must be positive real scalar counts or Inf"))
        }
    };
    if number == f64::INFINITY {
        return Ok(None);
    }
    let count = common::option_usize_value("optimwarmstart", Some(value), name, 0)?;
    if count == 0 {
        return Err(policy().invalid("memory limits must be positive or Inf"));
    }
    Ok(Some(count))
}
