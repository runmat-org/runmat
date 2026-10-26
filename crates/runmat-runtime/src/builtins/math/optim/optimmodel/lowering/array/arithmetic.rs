//! Affine scalar multiplication, retaining immutable coefficient/offset owners.
use super::super::terms::Term;
use super::*;
pub(super) async fn scale(
    input: Array,
    factor: f64,
    columns: usize,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Array> {
    if !factor.is_finite() {
        return Ok(Err(Stop::NonFinite));
    }
    if factor == 1. {
        return Ok(Ok(input));
    }
    let offset = match &input.offset {
        Offset::Uniform(value) => {
            let value = value * factor;
            if !value.is_finite() {
                return Ok(Err(Stop::NonFinite));
            }
            Offset::Uniform(value)
        }
        Offset::Dense(value) => {
            let result = evaluation::invoke(
                "times",
                &[Value::Tensor(value.clone()), Value::Num(factor)],
                policy,
                cooperation,
            )
            .await?;
            if !construct::finite_numeric(result.value(), policy, cooperation).await? {
                return Ok(Err(Stop::NonFinite));
            }
            match Offset::from_value(result.value(), policy)? {
                Ok(value) => value,
                Err(stop) => return Ok(Err(stop)),
            }
        }
        Offset::Sparse(value) => {
            let result = evaluation::invoke(
                "times",
                &[Value::SparseTensor(value.clone()), Value::Num(factor)],
                policy,
                cooperation,
            )
            .await?;
            if !construct::finite_numeric(result.value(), policy, cooperation).await? {
                return Ok(Err(Stop::NonFinite));
            }
            match Offset::from_value(result.value(), policy)? {
                Ok(value) => value,
                Err(stop) => return Ok(Err(stop)),
            }
        }
    };
    let coefficients = if factor == 0. || input.coefficients.is_zero() {
        Coefficients::Zero
    } else {
        let mut entries = runmat_value::exact_capacity_vec(input.coefficients.nonzeros())
            .map_err(|e| policy.storage(e))?;
        for term in input.coefficients.entries() {
            entries.push(Term {
                value: term.value * factor,
                ..term
            });
            cooperation.completed(1).await?;
        }
        match Coefficients::from_entries(
            entries,
            input.count(policy)?,
            columns,
            policy,
            cooperation,
        )
        .await?
        {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        }
    };
    Ok(Ok(Array {
        shape: input.shape,
        offset,
        coefficients,
        assigned: input.assigned,
        degree: input.degree,
    }))
}

pub(super) async fn add(
    left: Array,
    right: Array,
    operator: runmat_types::OperatorKind,
    shape: SharedShape,
    columns: usize,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Array> {
    use runmat_types::OperatorKind as Op;
    if !matches!(operator, Op::Add | Op::Subtract) {
        return Err(policy.invalid("compiled additive operation requires plus or minus"));
    }
    let plan = crate::builtins::common::broadcast::BroadcastPlan::new(&left.shape, &right.shape)
        .map_err(|source| policy.invalid(source))?;
    if !crate::builtins::common::shape::array_shapes_equal(plan.output_shape(), &shape) {
        return Err(policy.invalid("compiled broadcast differs from original expression geometry"));
    }
    let left_strides = crate::builtins::common::broadcast::compute_strides(&left.shape);
    let right_strides = crate::builtins::common::broadcast::compute_strides(&right.shape);
    let degree = Degree::collect(
        plan.len(),
        |row| {
            use crate::builtins::common::broadcast::broadcast_index;
            Ok(left
                .degree
                .at(broadcast_index(row, &shape, &left.shape, &left_strides))
                || right
                    .degree
                    .at(broadcast_index(row, &shape, &right.shape, &right_strides)))
        },
        policy,
        cooperation,
    )
    .await?;
    let mut left_rows = if left.count(policy)? != plan.len() {
        Some(runmat_value::exact_capacity_vec(plan.len()).map_err(|e| policy.storage(e))?)
    } else {
        None
    };
    let mut right_rows = if right.count(policy)? != plan.len() {
        Some(runmat_value::exact_capacity_vec(plan.len()).map_err(|e| policy.storage(e))?)
    } else {
        None
    };
    if left_rows.is_some() || right_rows.is_some() {
        for (_, a, b) in plan.iter() {
            if let Some(rows) = &mut left_rows {
                rows.push(a)
            }
            if let Some(rows) = &mut right_rows {
                rows.push(b)
            }
            cooperation.completed(1).await?;
        }
    }
    let left = match left_rows {
        Some(rows) => match left
            .gather(&rows, shape.clone(), columns, policy, cooperation)
            .await?
        {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        },
        None => Array {
            shape: shape.clone(),
            ..left
        },
    };
    let right = match right_rows {
        Some(rows) => match right
            .gather(&rows, shape.clone(), columns, policy, cooperation)
            .await?
        {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        },
        None => Array {
            shape: shape.clone(),
            ..right
        },
    };
    let offset = match (&left.offset, &right.offset) {
        (Offset::Uniform(a), Offset::Uniform(b)) => {
            let value = if operator == Op::Add { a + b } else { a - b };
            if !value.is_finite() {
                return Ok(Err(Stop::NonFinite));
            }
            Offset::Uniform(value)
        }
        (_, Offset::Uniform(value)) if *value == 0. => left.offset.clone(),
        (Offset::Uniform(value), _) if *value == 0. && operator == Op::Add => right.offset.clone(),
        _ => {
            let result = evaluation::invoke(
                operator
                    .overload_name()
                    .expect("checked additive operator")
                    .display_name(),
                &[left.offset.value(), right.offset.value()],
                policy,
                cooperation,
            )
            .await?;
            if !construct::finite_numeric(result.value(), policy, cooperation).await? {
                return Ok(Err(Stop::NonFinite));
            }
            match Offset::from_value(result.value(), policy)? {
                Ok(value) => value,
                Err(stop) => return Ok(Err(stop)),
            }
        }
    };
    let assigned = if matches!(left.assigned, Assigned::Missing)
        || matches!(right.assigned, Assigned::Missing)
    {
        Assigned::Missing
    } else if matches!(left.assigned, Assigned::All) && matches!(right.assigned, Assigned::All) {
        Assigned::All
    } else {
        let mut mask =
            runmat_value::exact_capacity_vec(plan.len()).map_err(|e| policy.storage(e))?;
        let mut missing = false;
        for row in 0..plan.len() {
            let valid = left.assigned.at(row, policy)? && right.assigned.at(row, policy)?;
            mask.push(valid);
            missing |= !valid;
            cooperation.completed(1).await?;
        }
        if missing {
            Assigned::Partial(std::rc::Rc::new(mask))
        } else {
            Assigned::All
        }
    };
    let coefficients = if right.coefficients.is_zero() {
        left.coefficients
    } else if left.coefficients.is_zero() && operator == Op::Add {
        right.coefficients
    } else {
        let capacity = left
            .coefficients
            .nonzeros()
            .checked_add(right.coefficients.nonzeros())
            .ok_or_else(|| policy.invalid("compiled additive coefficient count overflow"))?;
        let mut entries =
            runmat_value::exact_capacity_vec(capacity).map_err(|e| policy.storage(e))?;
        for term in left.coefficients.entries() {
            entries.push(term);
            cooperation.completed(1).await?;
        }
        for mut term in right.coefficients.entries() {
            if operator == Op::Subtract {
                term.value = -term.value
            }
            entries.push(term);
            cooperation.completed(1).await?;
        }
        let result =
            match Coefficients::from_entries(entries, plan.len(), columns, policy, cooperation)
                .await?
            {
                Ok(value) => value,
                Err(stop) => return Ok(Err(stop)),
            };
        result
    };
    Ok(Ok(Array {
        shape,
        offset,
        coefficients,
        assigned,
        degree,
    }))
}
