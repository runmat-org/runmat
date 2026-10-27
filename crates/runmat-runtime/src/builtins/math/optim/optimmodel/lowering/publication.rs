//! Original variable/constraint presentations over one accepted point owner.
//! Backend workspace is already gone; no callbacks or graph evaluation replay.
use super::*;
use crate::execution::{RootedValue, RootedValueSequence};
pub(super) mod information;
pub(super) mod multipliers;
mod stopped;
pub(in crate::builtins::math::optim::optimmodel) async fn publish(
    solved: Solved,
    requested: Option<usize>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValueSequence> {
    match solved {
        Solved::Left(solved) => algebraic(solved, requested, policy, cooperation).await,
        Solved::Right(futures::future::Either::Left(NonlinearSolved {
            solved,
            formulation,
        })) => nonlinear::publish(solved, &formulation, requested, policy, cooperation).await,
        Solved::Right(futures::future::Either::Right(solved)) => {
            stopped::publish(solved, requested, policy, cooperation).await
        }
    }
}
async fn algebraic(
    mut solved: AlgebraicSolved,
    requested: Option<usize>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValueSequence> {
    let flag = solved.outcome.flag(policy)?;
    let count = requested.unwrap_or(1);
    if count > 5 {
        return Err(policy.invalid("solve returns at most five outputs"));
    }
    let mut roots = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    if count > 0 {
        let solution = match &solved.point {
            Some(point) => RootedValue::in_optional_context(
                point.solution.value().clone(),
                cooperation.runtime(),
            )?,
            None => empty_solution(&solved.program.layout, policy, cooperation).await?,
        };
        roots.push(solution);
    }
    if count > 1 {
        let value = match (&solved.point, &solved.formulation.objective) {
            (Some(point), Some(objective)) => RootedValue::in_optional_context(
                point.root(objective.root, policy)?.value().clone(),
                cooperation.runtime(),
            )?,
            _ => empty(policy, cooperation)?,
        };
        roots.push(value);
    }
    if count > 2 {
        roots.push(super::super::exitflag::publish(
            flag,
            policy,
            cooperation.runtime(),
        )?);
    }
    if count > 3 {
        roots.push(information::publish(&mut solved.outcome, policy, cooperation).await?);
    }
    if count > 4 {
        roots.push(multipliers::publish(&mut solved, policy, cooperation).await?);
    }
    finish(roots, requested, policy, cooperation).await
}
pub(super) async fn finish(
    roots: Vec<RootedValue>,
    requested: Option<usize>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValueSequence> {
    let mut values =
        runmat_value::exact_capacity_vec(roots.len()).map_err(|e| policy.storage(e))?;
    for value in &roots {
        values.push(value.value().clone());
        cooperation.completed(1).await?;
    }
    let values = if requested.is_none() {
        runmat_value::ValueSequence::single(values.remove(0))
    } else {
        runmat_value::ValueSequence::comma_separated(values)
    }
    .map_err(crate::sequence::sequence_error_to_runtime)?;
    let output = match cooperation.runtime() {
        Some(runtime) => RootedValueSequence::in_context(values, runtime),
        None => crate::context::with_original_scope(None, || {
            RootedValueSequence::new(values).map_err(|e| e.into_runtime_error())
        }),
    }?;
    drop(roots);
    Ok(output)
}
pub(super) fn empty(policy: Policy, cooperation: &ScalarCooperation) -> BuiltinResult<RootedValue> {
    let value = Value::Tensor(
        runmat_value::Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| policy.invalid(e))?,
    );
    RootedValue::in_optional_context(value, cooperation.runtime())
}
pub(super) async fn empty_solution(
    layout: &layout::Layout,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let empty = empty(policy, cooperation)?;
    let mut fields =
        runmat_value::exact_capacity_vec(layout.order.len()).map_err(|e| policy.storage(e))?;
    for &index in &layout.order {
        fields.push((layout.variables[index].name.as_str(), empty.value()));
        cooperation.completed(1).await?;
    }
    crate::execution::structure::construct_struct_from_fields_in_optional_context(
        cooperation.runtime(),
        fields.into_iter(),
    )
}
