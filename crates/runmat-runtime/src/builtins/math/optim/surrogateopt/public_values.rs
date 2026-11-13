//! Public projections read the one measured generation. Array storage is
//! allocated only for a requested output/observer, never retained as a mirror.
use super::{
    history::{History, Trial},
    invalid,
    observation::Objective,
};
use crate::{builtins::math::optim::common::ScalarCooperation, BuiltinResult};
use runmat_value::{Tensor, Value};

pub(super) fn empty(c: &ScalarCooperation) -> BuiltinResult<Value> {
    crate::builtins::math::optim::common::output_callback::empty(c.runtime())
}
pub(super) fn objective(trial: Option<&Trial>, c: &ScalarCooperation) -> BuiltinResult<Value> {
    match trial.map(|t| t.observation.objective()) {
        Some(Objective::Measured(value)) => Ok(Value::Num(value)),
        Some(Objective::Absent) | None => empty(c),
    }
}
pub(super) fn origin(trial: Option<&Trial>, c: &ScalarCooperation) -> BuiltinResult<Value> {
    let Some(trial) = trial else { return empty(c) };
    let text = trial.origin.name();
    crate::call::argument_text::copy_str(text, c.runtime()).map(|text| Value::String(text.into()))
}
pub(super) fn violation(trial: Option<&Trial>, c: &ScalarCooperation) -> BuiltinResult<Value> {
    match trial {
        Some(trial) => Ok(Value::Num(trial.observation.maximum().unwrap_or(0.0))),
        None => empty(c),
    }
}
pub(super) fn capacity<T>(count: usize) -> BuiltinResult<Vec<T>> {
    runmat_value::exact_capacity_vec(count).map_err(|source| {
        crate::build_runtime_error("surrogateopt: public output allocation failed")
            .with_builtin(super::NAME)
            .with_source(source)
            .build()
    })
}
pub(super) fn tensor(
    values: Vec<f64>,
    rows: usize,
    columns: usize,
    c: &ScalarCooperation,
) -> BuiltinResult<Value> {
    // Public solve ingress rejects finite workspace before any effects. Keep
    // this leaf safe when called independently by a future output consumer.
    crate::context::host_allocation::require_unbounded_workspace(
        c.runtime(),
        super::NAME,
        "finite host-memory execution has no complete surrogate output bound",
    )?;
    if rows.checked_mul(columns) != Some(values.len()) {
        return Err(invalid("public output changed its measured extent"));
    }
    let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
    Tensor::new(values, vec![rows, columns])
        .map(Value::Tensor)
        .map_err(|error| invalid(format!("invalid public output storage: {error}")))
}
pub(super) async fn point(
    trial: Option<&Trial>,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    let Some(trial) = trial else { return empty(c) };
    let n = trial.point.variables();
    let mut values = capacity(n)?;
    for column in 0..n {
        values.push(trial.point.coordinate(column));
        c.completed(1).await?;
    }
    tensor(values, 1, n, c)
}
pub(super) async fn inequalities(
    trial: Option<&Trial>,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    let Some(trial) = trial else { return empty(c) };
    let columns = trial.observation.schema().inequalities;
    let mut values = capacity(columns)?;
    for column in 0..columns {
        values.push(trial.observation.inequality(column));
        c.completed(1).await?;
    }
    tensor(values, 1, columns, c)
}
pub(super) async fn trials(history: &History, c: &mut ScalarCooperation) -> BuiltinResult<Value> {
    use super::{history::TrialIndex, observation::ObjectiveKind};
    let rows = history.len();
    let n = history.variables();
    let q = history.schema().map_or(0, |s| s.inequalities);
    let mut x = capacity(
        rows.checked_mul(n)
            .ok_or_else(|| invalid("trial point extent overflow"))?,
    )?;
    for column in 0..n {
        for row in 0..rows {
            x.push(history.trial(TrialIndex(row)).point.coordinate(column));
            c.completed(1).await?;
        }
    }
    let mut ineq = capacity(
        rows.checked_mul(q)
            .ok_or_else(|| invalid("trial inequality extent overflow"))?,
    )?;
    for column in 0..q {
        for row in 0..rows {
            ineq.push(
                history
                    .trial(TrialIndex(row))
                    .observation
                    .inequality(column),
            );
            c.completed(1).await?;
        }
    }
    let fval = if history
        .schema()
        .is_some_and(|s| s.objective == ObjectiveKind::Present)
    {
        let mut values = capacity(rows)?;
        for row in 0..rows {
            let Objective::Measured(value) = history.trial(TrialIndex(row)).observation.objective()
            else {
                return Err(invalid("trial objective changed its measured schema"));
            };
            values.push(value);
            c.completed(1).await?;
        }
        tensor(values, rows, 1, c)?
    } else {
        empty(c)?
    };
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("X", tensor(x, rows, n, c)?);
    fields.insert("Fval", fval);
    fields.insert("Ineq", tensor(ineq, rows, q, c)?);
    Ok(Value::Struct(fields.finish()))
}
