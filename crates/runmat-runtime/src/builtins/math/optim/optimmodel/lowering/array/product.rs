//! Elementwise affine products use the canonical implicit-expansion geometry.
//! Constants retain their real backing; only active decision coefficients are
//! projected and weighted. Dependence is structural, never inferred from zeros.
use super::super::terms::Term;
use super::*;
use crate::builtins::common::broadcast::{broadcast_index, compute_strides, BroadcastPlan};
pub(super) async fn multiply(
    left: Array,
    right: Array,
    shape: SharedShape,
    columns: usize,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Array> {
    let plan =
        BroadcastPlan::new(&left.shape, &right.shape).map_err(|source| policy.invalid(source))?;
    if !crate::builtins::common::shape::array_shapes_equal(plan.output_shape(), &shape) {
        return Err(policy.invalid("compiled product differs from original broadcast geometry"));
    }
    for (_, a, b) in plan.iter() {
        if left.degree.at(a) && right.degree.at(b) {
            return Ok(Err(Stop::NonAffine));
        }
        cooperation.completed(1).await?;
    }
    let a_strides = compute_strides(&left.shape);
    let b_strides = compute_strides(&right.shape);
    let degree = Degree::collect(
        plan.len(),
        |row| {
            Ok(left
                .degree
                .at(broadcast_index(row, &shape, &left.shape, &a_strides))
                || right
                    .degree
                    .at(broadcast_index(row, &shape, &right.shape, &b_strides)))
        },
        policy,
        cooperation,
    )
    .await?;
    let offset = match (&left.offset, &right.offset) {
        (Offset::Uniform(a), Offset::Uniform(b)) => {
            let value = a * b;
            if !value.is_finite() {
                return Ok(Err(Stop::NonFinite));
            }
            Offset::Uniform(value)
        }
        (Offset::Uniform(a), _) | (_, Offset::Uniform(a)) if *a == 0. => Offset::Uniform(0.),
        _ => {
            let mut values = runmat_value::exact_capacity_vec(plan.len())
                .map_err(|source| policy.storage(source))?;
            for (_, a, b) in plan.iter() {
                let value = left.offset.at(a, policy)? * right.offset.at(b, policy)?;
                if !value.is_finite() {
                    return Ok(Err(Stop::NonFinite));
                }
                values.push(value);
                cooperation.completed(1).await?;
            }
            Offset::Dense(
                Tensor::from_numeric_storage(
                    runmat_value::NumericStorage::F64(values),
                    shape.clone(),
                )
                .map_err(|source| policy.invalid(source))?,
            )
        }
    };
    let assigned =
        if matches!(left.assigned, Assigned::All) && matches!(right.assigned, Assigned::All) {
            Assigned::All
        } else {
            let mut values = runmat_value::exact_capacity_vec(plan.len())
                .map_err(|source| policy.storage(source))?;
            let mut missing = false;
            for (_, a, b) in plan.iter() {
                let valid = left.assigned.at(a, policy)? && right.assigned.at(b, policy)?;
                missing |= !valid;
                values.push(valid);
                cooperation.completed(1).await?;
            }
            if missing {
                Assigned::Partial(std::rc::Rc::new(values))
            } else {
                Assigned::All
            }
        };
    let mut entries = Vec::new();
    for (affine, constant, is_left) in [(&left, &right, true), (&right, &left, false)] {
        if affine.coefficients.is_zero() {
            continue;
        }
        let map = if affine.count(policy)? == plan.len() {
            projection::RowMap::Identity { rows: plan.len() }
        } else {
            let mut rows =
                runmat_value::exact_capacity_vec(plan.len()).map_err(|e| policy.storage(e))?;
            for (_, a, b) in plan.iter() {
                rows.push(if is_left { a } else { b });
                cooperation.completed(1).await?;
            }
            match projection::RowMap::gather(&rows, affine.count(policy)?, policy, cooperation)
                .await?
            {
                Ok(map) => map,
                Err(stop) => return Ok(Err(stop)),
            }
        };
        let strides = compute_strides(&constant.shape);
        for term in affine.coefficients.entries() {
            for (row, weight) in map.entries(term.row) {
                let index = broadcast_index(row, &shape, &constant.shape, &strides);
                let value = term.value * weight * constant.offset.at(index, policy)?;
                if !value.is_finite() {
                    return Ok(Err(Stop::NonFinite));
                }
                if value != 0. {
                    entries.try_reserve(1).map_err(|e| policy.reserve(e))?;
                    entries.push(Term {
                        column: term.column,
                        row,
                        value,
                    });
                }
                cooperation.completed(1).await?;
            }
        }
    }
    let coefficients = match Coefficients::from_entries(
        entries,
        plan.len(),
        columns,
        policy,
        cooperation,
    )
    .await?
    {
        Ok(values) => values,
        Err(stop) => return Ok(Err(stop)),
    };
    Ok(Ok(Array {
        shape,
        offset,
        coefficients,
        assigned,
        degree,
    }))
}
