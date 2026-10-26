//! Original-state outputs: no callback replay or multiplier role conflation.
use super::{model::Geometry, outcome::Outcome, workspace};
use crate::builtins::math::optim::common::{self, solver_timing::SolverTiming, ScalarCooperation};
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};
#[derive(Clone, Copy)]
pub(super) struct ReflectiveInfo {
    pub cg_iterations: usize,
    pub radius: f64,
}
pub(super) mod information;
enum FinalHessian {
    Approximate(Option<clarabel::algebra::CscMatrix<f64>>),
    Retained(Value),
}

fn column(values: Vec<f64>) -> BuiltinResult<Value> {
    let n = values.len();
    Tensor::new(values, vec![n, 1])
        .map(Value::Tensor)
        .map_err(super::invalid)
}
pub(super) async fn finalize(
    mut outcome: Outcome,
    geometry: &Geometry,
    shape: &[usize],
    scalar: bool,
    timing: Option<SolverTiming>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let hessian = FinalHessian::Approximate(outcome.hessian.take());
    finalize_values(
        outcome,
        geometry,
        shape,
        scalar,
        timing,
        hessian,
        None,
        cooperation,
    )
    .await
}
pub(super) async fn finalize_reflective(
    outcome: Outcome,
    geometry: &Geometry,
    shape: &[usize],
    scalar: bool,
    timing: Option<SolverTiming>,
    model: Option<crate::execution::RootedValue>,
    info: ReflectiveInfo,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    if outcome.hessian.is_some() {
        return Err(super::invalid(
            "reflective publication has conflicting model owners",
        ));
    }
    if let Some(model) = model {
        model
            .with_owned_async(|value| async move {
                // The successor sequence is registered while the original
                // H/Hinfo root and every reachable descendant are still retained.
                finalize_values(
                    outcome,
                    geometry,
                    shape,
                    scalar,
                    timing,
                    FinalHessian::Retained(value),
                    Some(info),
                    cooperation,
                )
                .await
            })
            .await
    } else {
        finalize_values(
            outcome,
            geometry,
            shape,
            scalar,
            timing,
            FinalHessian::Approximate(None),
            Some(info),
            cooperation,
        )
        .await
    }
}
async fn finalize_values(
    mut outcome: Outcome,
    geometry: &Geometry,
    shape: &[usize],
    scalar: bool,
    timing: Option<SolverTiming>,
    hessian: FinalHessian,
    reflective: Option<ReflectiveInfo>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let requested = crate::output_count::current_output_count_in(cooperation.runtime());
    let count = requested.unwrap_or(1);
    let mut values = workspace::capacity(count, super::IDENTITY)?;
    let information = if count >= 4 {
        Some(information::value(&outcome, timing.as_ref(), reflective)?)
    } else {
        None
    };
    if count > 0 {
        values.push(common::vector_to_value(
            "fmincon", outcome.x, shape, scalar,
        )?);
    }
    if count >= 2 {
        values.push(
            outcome
                .point
                .as_ref()
                .map(|value| Value::Num(value.f))
                .unwrap_or_else(super::empty),
        );
    }
    if count >= 3 {
        values.push(Value::Num(outcome.termination.flag() as f64));
    }
    if let Some(information) = information {
        values.push(information);
    }
    if count >= 5 {
        if let Some(constraints) = outcome
            .constraints
            .filter(|_| outcome.multipliers_available)
        {
            if constraints.roles.len() != outcome.multipliers.len() {
                return Err(super::invalid("multiplier role count is inconsistent"));
            }
            let groups = super::validation::split(
                &constraints,
                &outcome.multipliers,
                geometry.variables,
                geometry.inequality.rhs.len(),
                geometry.equality.rhs.len(),
                outcome
                    .point
                    .as_ref()
                    .map_or(0, |point| point.inequality.len()),
                outcome
                    .point
                    .as_ref()
                    .map_or(0, |point| point.equality.len()),
                super::IDENTITY,
                cooperation,
            )
            .await?;
            let mut fields = runmat_value::UnboundedStructValueBuilder::new();
            for (name, group) in [
                "ineqlin",
                "eqlin",
                "ineqnonlin",
                "eqnonlin",
                "lower",
                "upper",
            ]
            .into_iter()
            .zip([
                groups.ineqlin,
                groups.eqlin,
                groups.ineqnonlin,
                groups.eqnonlin,
                groups.lower,
                groups.upper,
            ]) {
                fields.insert(name, column(group)?);
            }
            values.push(Value::Struct(fields.finish()));
        } else {
            values.push(super::empty());
        }
    }
    if count >= 6 {
        values.push(
            match outcome
                .point
                .as_mut()
                .and_then(|point| point.gradient.take())
            {
                Some(mut gradient) => {
                    if outcome
                        .point
                        .as_ref()
                        .is_some_and(|point| point.projected_objective)
                    {
                        for ((value, lower), upper) in gradient
                            .iter_mut()
                            .zip(&geometry.lower)
                            .zip(&geometry.upper)
                        {
                            if lower == upper {
                                *value = f64::NAN;
                            }
                            cooperation.completed(1).await?;
                        }
                    }
                    column(gradient)?
                }
                None => super::empty(),
            },
        );
    }
    if count >= 7 {
        values.push(match hessian {
            FinalHessian::Approximate(Some(matrix)) => {
                let count = matrix
                    .n
                    .checked_mul(matrix.n)
                    .ok_or_else(|| super::invalid("Hessian output extent overflow"))?;
                let mut entries = zeros(count, cooperation).await?;
                for col in 0..matrix.n {
                    for index in matrix.colptr[col]..matrix.colptr[col + 1] {
                        let row = matrix.rowval[index];
                        let value = matrix.nzval[index];
                        entries[col * matrix.n + row] = value;
                        entries[row * matrix.n + col] = value;
                        cooperation.completed(1).await?;
                    }
                }
                Value::Tensor(
                    Tensor::new(entries, vec![matrix.n, matrix.n]).map_err(super::invalid)?,
                )
            }
            FinalHessian::Approximate(None) => super::empty(),
            FinalHessian::Retained(value) => value,
        });
    }
    let sequence = if requested.is_none() {
        runmat_value::ValueSequence::single(values.remove(0))
    } else {
        runmat_value::ValueSequence::comma_separated(values)
    }
    .map_err(crate::sequence::sequence_error_to_runtime)?;
    match cooperation.runtime() {
        Some(runtime) => crate::execution::RootedValueSequence::in_context(sequence, runtime),
        None => crate::execution::RootedValueSequence::new(sequence)
            .map_err(|source| source.into_runtime_error()),
    }
}
async fn zeros(length: usize, cooperation: &mut ScalarCooperation) -> BuiltinResult<Vec<f64>> {
    let mut values = workspace::capacity(length, super::IDENTITY)?;
    for _ in 0..length {
        values.push(0.0);
        cooperation.completed(1).await?;
    }
    Ok(values)
}
