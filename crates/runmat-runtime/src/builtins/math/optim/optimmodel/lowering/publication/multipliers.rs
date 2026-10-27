//! Consume solver dual vectors into the actual original shaped output fields.
//! No cancelled row/variable is removed from the public role layout.
use super::*;
use runmat_value::{NumericStorage, Tensor};
async fn take(
    values: &mut std::vec::IntoIter<f64>,
    count: usize,
    shape: SharedShape,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let mut data = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    for _ in 0..count {
        data.push(
            values
                .next()
                .ok_or_else(|| policy.invalid("solver dual is shorter than its original role"))?,
        );
        cooperation.completed(1).await?;
    }
    RootedValue::in_optional_context(
        Value::Tensor(
            Tensor::from_numeric_storage(NumericStorage::F64(data), shape)
                .map_err(|e| policy.invalid(e))?,
        ),
        cooperation.runtime(),
    )
}
pub(super) async fn publish(
    solved: &mut AlgebraicSolved,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if matches!(solved.outcome, report::Outcome::Integer(_)) {
        return empty(policy, cooperation);
    }
    project(
        &solved.program.layout,
        &solved.formulation,
        &solved.rows,
        solved.outcome.multipliers(),
        policy,
        cooperation,
    )
    .await
}
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn project(
    layout: &layout::Layout,
    formulation: &formulation::Formulation,
    rows: &[assembly::Rows],
    multiplier: Option<report::Multipliers>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let valid = multiplier.is_some();
    let (mut lower, mut upper, mut equality, mut inequality) = match multiplier {
        Some(multiplier) => (
            multiplier.lower.into_iter(),
            multiplier.upper.into_iter(),
            multiplier.eqlin.into_iter(),
            multiplier.ineqlin.into_iter(),
        ),
        None => (
            Vec::new().into_iter(),
            Vec::new().into_iter(),
            Vec::new().into_iter(),
            Vec::new().into_iter(),
        ),
    };
    let mut variables =
        runmat_value::exact_capacity_vec(layout.variables.len()).map_err(|e| policy.storage(e))?;
    for &index in &layout.order {
        let variable = &layout.variables[index];
        let a = if valid {
            take(
                &mut lower,
                variable.count,
                variable.shape.clone(),
                policy,
                cooperation,
            )
            .await?
        } else {
            empty(policy, cooperation)?
        };
        let b = if valid {
            take(
                &mut upper,
                variable.count,
                variable.shape.clone(),
                policy,
                cooperation,
            )
            .await?
        } else {
            empty(policy, cooperation)?
        };
        let fields = crate::execution::structure::construct_struct_from_fields_in_optional_context(
            cooperation.runtime(),
            [("Lower", a.value()), ("Upper", b.value())].into_iter(),
        )?;
        variables.push((variable.name.as_str(), fields));
    }
    let variables = crate::execution::structure::construct_struct_from_fields_in_optional_context(
        cooperation.runtime(),
        variables.iter().map(|(name, value)| (*name, value.value())),
    )?;
    let mut constraints =
        runmat_value::exact_capacity_vec(rows.len()).map_err(|e| policy.storage(e))?;
    for (constraint, row) in formulation.constraints.iter().zip(rows.iter()) {
        let value = if valid {
            take(
                if row.equality {
                    &mut equality
                } else {
                    &mut inequality
                },
                row.count,
                constraint.shape.clone(),
                policy,
                cooperation,
            )
            .await?
        } else {
            empty(policy, cooperation)?
        };
        constraints.push((constraint.name.as_ref().map(Utf8Text::as_str), value));
    }
    if valid
        && (!lower.as_slice().is_empty()
            || !upper.as_slice().is_empty()
            || !equality.as_slice().is_empty()
            || !inequality.as_slice().is_empty())
    {
        return Err(policy.invalid("solver dual is longer than its original role layout"));
    }
    let constraints = if constraints.len() == 1 && constraints[0].0.is_none() {
        constraints.pop().expect("one unnamed constraint").1
    } else {
        if constraints.iter().any(|(name, _)| name.is_none()) {
            return Err(policy.invalid("mixed named and unnamed constraint roles"));
        }
        crate::execution::structure::construct_struct_from_fields_in_optional_context(
            cooperation.runtime(),
            constraints
                .iter()
                .map(|(name, value)| (name.expect("named collection"), value.value())),
        )?
    };
    crate::execution::structure::construct_struct_from_fields_in_optional_context(
        cooperation.runtime(),
        [
            ("Variables", variables.value()),
            ("Constraints", constraints.value()),
        ]
        .into_iter(),
    )
}
