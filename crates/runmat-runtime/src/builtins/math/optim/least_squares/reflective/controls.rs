//! Only the effective reflective method inspects its live controls.
use super::*;
use crate::builtins::math::optim::common::{self, ScalarOptionFields};
use runmat_builtins::{OptimOptionField as Field, OptimSubproblemAlgorithm};
use runmat_value::{StructValue, Value};
#[derive(Clone, Copy)]
pub(super) enum Subproblem {
    Factorization,
    Cg {
        bandwidth: usize,
        maximum: usize,
        tolerance: f64,
    },
}
pub(super) struct Controls {
    pub criteria: Criteria,
    pub subproblem: Subproblem,
}
pub(super) async fn prepare(
    options: &LeastSquaresOptions,
    source: Option<&StructValue>,
    variables: usize,
    evaluator: &mut impl LeastSquaresEvaluator,
) -> BuiltinResult<Controls> {
    let fields =
        ScalarOptionFields::from_struct_for_solver(source, runmat_builtins::OptimSolver::Lsqnonlin);
    if let Some(value) = fields.exact(Field::SubproblemAlgorithm) {
        let text = match value {
            Value::String(runmat_value::StringElement::Text(_)) => true,
            Value::StringArray(value) => value.data().len() == 1,
            Value::CharArray(value) => value.rows == 1,
            _ => false,
        };
        if !text {
            return Err(invalid(
                "SubproblemAlgorithm must be a character vector or string scalar".into(),
            ));
        }
    }
    let method =
        OptimSubproblemAlgorithm::parse(&fields.text(Field::SubproblemAlgorithm, "factorization")?)
            .ok_or_else(|| invalid("SubproblemAlgorithm must be cg or factorization".into()))?;
    // Bandwidth is a role selector: infinity changes CG to direct before any
    // otherwise dormant PCG carrier is inspected.
    let selected_bandwidth = if method == OptimSubproblemAlgorithm::Cg {
        bandwidth(&fields, evaluator).await?
    } else {
        None
    };
    let mut live = [None; 3];
    if options.criteria.is_some() {
        live[0] = Some(Field::OptimalityTolerance);
    }
    if selected_bandwidth.is_some() {
        live[1] = Some(Field::MaxPCGIter);
        live[2] = Some(Field::TolPCG);
    }
    for field in live.into_iter().flatten() {
        if let Some(value) = fields.exact(field) {
            preflight(value, field, evaluator)?;
        }
    }
    let mut criteria = options.criteria(Method::Reflective);
    if options.criteria.is_some() {
        criteria.optimality = real(
            &fields,
            Field::OptimalityTolerance,
            criteria.optimality,
            evaluator,
        )
        .await?;
        if criteria.optimality < 0. {
            return Err(invalid("OptimalityTolerance must be nonnegative".into()));
        }
        if fields.exact(Field::TolFun).is_some()
            && fields.exact(Field::OptimalityTolerance).is_some()
            && criteria.optimality != options.criteria(Method::Reflective).optimality
        {
            return Err(invalid(
                "conflicting TolFun and OptimalityTolerance controls".into(),
            ));
        }
    }
    let subproblem = match selected_bandwidth {
        None => Subproblem::Factorization,
        Some(bandwidth) => {
            let maximum = count(
                &fields,
                Field::MaxPCGIter,
                (variables / 2).max(1),
                evaluator,
            )
            .await?;
            let tolerance = real(&fields, Field::TolPCG, 0.1, evaluator).await?;
            if maximum == 0 || !(tolerance > 0. && tolerance.is_finite()) {
                return Err(invalid("MaxPCGIter and TolPCG must be positive".into()));
            }
            Subproblem::Cg {
                bandwidth,
                maximum,
                tolerance,
            }
        }
    };
    Ok(Controls {
        criteria,
        subproblem,
    })
}
fn preflight(
    value: &Value,
    field: Field,
    evaluator: &mut impl LeastSquaresEvaluator,
) -> BuiltinResult<()> {
    common::preflight_option_scalar("lsqnonlin", value, field)?;
    common::preflight_host_residency("lsqnonlin", value, None, cooperation(evaluator)?.runtime())
}

async fn materialize(
    value: &Value,
    field: Field,
    evaluator: &mut impl LeastSquaresEvaluator,
) -> BuiltinResult<Option<Value>> {
    preflight(value, field, evaluator)?;
    let c = cooperation(evaluator)?;
    if !matches!(value, Value::GpuTensor(_)) {
        return Ok(None);
    }
    let result = crate::dispatcher::gather_if_needed_in_context_async(value, c.runtime()).await?;
    c.completed(0).await?;
    Ok(Some(result))
}
async fn real(
    fields: &ScalarOptionFields<'_>,
    field: Field,
    default: f64,
    evaluator: &mut impl LeastSquaresEvaluator,
) -> BuiltinResult<f64> {
    let Some(value) = fields.exact(field) else {
        return Ok(default);
    };
    let gathered = materialize(value, field, evaluator).await?;
    common::option_f64_value(
        "lsqnonlin",
        Some(gathered.as_ref().unwrap_or(value)),
        field.name(),
        default,
    )
}
async fn count(
    fields: &ScalarOptionFields<'_>,
    field: Field,
    default: usize,
    evaluator: &mut impl LeastSquaresEvaluator,
) -> BuiltinResult<usize> {
    let Some(value) = fields.exact(field) else {
        return Ok(default);
    };
    let gathered = materialize(value, field, evaluator).await?;
    common::option_usize_value(
        "lsqnonlin",
        Some(gathered.as_ref().unwrap_or(value)),
        field.name(),
        default,
    )
}
async fn bandwidth(
    fields: &ScalarOptionFields<'_>,
    evaluator: &mut impl LeastSquaresEvaluator,
) -> BuiltinResult<Option<usize>> {
    let Some(value) = fields.exact(Field::PrecondBandWidth) else {
        return Ok(None);
    };
    let gathered = materialize(value, Field::PrecondBandWidth, evaluator).await?;
    let value = gathered.as_ref().unwrap_or(value);
    let infinite = match value {
        Value::Num(value) => *value == f64::INFINITY,
        Value::Tensor(value) if value.len() == 1 && value.integer_storage().is_none() => {
            crate::builtins::common::tensor::tensor_value_f64(value, 0) == f64::INFINITY
        }
        _ => false,
    };
    if infinite {
        return Ok(None);
    }
    common::option_usize_value("lsqnonlin", Some(value), Field::PrecondBandWidth.name(), 0)
        .map(Some)
}
