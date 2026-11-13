//! Integer constant powers and zero/identity variable powers. Exponents have
//! already been authenticated against their original exact source DAG.
use super::super::{
    array::Assigned,
    terms::{Coefficients, Term},
};
use super::*;
use runmat_types::OperatorKind as Op;
mod matrix;
pub(in crate::builtins::math::optim::optimmodel::lowering) use matrix::identity;
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn lower(
    mut left: Array,
    right: Array,
    operator: Op,
    exponents: &Exponents,
    shape: SharedShape,
    columns: usize,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Array> {
    if operator == Op::MatrixPower {
        if exponents.count(policy)? != 1 {
            return Ok(Err(Stop::UnsupportedFunction));
        }
        if left.count(policy)? != 1 {
            return matrix::lower(left, exponents.at(0, policy)?, shape, columns, policy, c).await;
        }
    }
    let plan = crate::builtins::common::broadcast::BroadcastPlan::new(&left.shape, &right.shape)
        .map_err(|e| policy.invalid(e))?;
    if !crate::builtins::common::shape::array_shapes_equal(plan.output_shape(), &shape) {
        return Err(policy.invalid("power broadcast differs from original shape"));
    }
    for (row, a, _) in plan.iter() {
        let exponent = exponents.broadcast(row, &shape, policy)?;
        if left.degree.at(a) && exponent != 0. && exponent != 1. {
            return Ok(Err(Stop::NonAffine));
        }
        c.completed(1).await?;
    }
    let count = plan.len();
    let mut all_zero = true;
    let mut all_identity = true;
    let mut uniform = true;
    let mut first = None;
    // Inspect the original immutable source without an expanded numeric view.
    for (row, a, _) in plan.iter() {
        if !left.assigned.at(a, policy)? {
            return Ok(Err(Stop::Unassigned));
        }
        let exponent = exponents.broadcast(row, &shape, policy)?;
        all_zero &= exponent == 0.;
        all_identity &= exponent == 1.;
        let value = left.offset.at(a, policy)?.powf(exponent);
        if !value.is_finite() {
            return Ok(Err(Stop::NonFinite));
        }
        if let Some(previous) = first {
            uniform &= value == previous;
        } else {
            first = Some(value);
        }
        c.completed(1).await?;
    }
    if all_zero {
        return Ok(Ok(Array {
            shape,
            offset: Offset::Uniform(1.),
            degree: Degree::Constant,
            coefficients: Coefficients::Zero,
            assigned: Assigned::All,
        }));
    }
    if all_identity && left.count(policy)? == count {
        left.shape = shape;
        return Ok(Ok(left));
    }
    let strides = crate::builtins::common::broadcast::compute_strides(&left.shape);
    let degree = Degree::collect(
        count,
        |row| {
            let source = crate::builtins::common::broadcast::broadcast_index(
                row,
                &shape,
                &left.shape,
                &strides,
            );
            Ok(exponents.broadcast(row, &shape, policy)? == 1. && left.degree.at(source))
        },
        policy,
        c,
    )
    .await?;
    let offset = if uniform {
        Offset::Uniform(first.unwrap_or(1.))
    } else {
        // Only genuinely nonuniform results own a numeric destination.
        let mut values = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
        for (row, a, _) in plan.iter() {
            values.push(
                left.offset
                    .at(a, policy)?
                    .powf(exponents.broadcast(row, &shape, policy)?),
            );
            c.completed(1).await?;
        }
        Offset::Dense(
            runmat_value::Tensor::from_numeric_storage(
                runmat_value::NumericStorage::F64(values),
                shape.clone(),
            )
            .map_err(|e| policy.invalid(e))?,
        )
    };
    let coefficients = if left.coefficients.is_zero() {
        Coefficients::Zero
    } else {
        let mut entries = Vec::new();
        if left.count(policy)? == count {
            for term in left.coefficients.entries() {
                if exponents.broadcast(term.row, &shape, policy)? == 1. {
                    entries.try_reserve(1).map_err(|e| policy.reserve(e))?;
                    entries.push(term);
                }
                c.completed(1).await?;
            }
        } else {
            // Coefficient projection uses the canonical broadcast indices only;
            // it does not gather a numeric copy of the source array.
            let mut routes = Vec::new();
            for (row, a, _) in plan.iter() {
                if exponents.broadcast(row, &shape, policy)? == 1. && left.degree.at(a) {
                    routes.try_reserve(1).map_err(|e| policy.reserve(e))?;
                    routes.push(Term {
                        column: a,
                        row,
                        value: 1.,
                    });
                }
                c.completed(1).await?;
            }
            let map =
                match Coefficients::from_entries(routes, count, left.count(policy)?, policy, c)
                    .await?
                {
                    Ok(value) => value,
                    Err(stop) => return Ok(Err(stop)),
                };
            for term in left.coefficients.entries() {
                for (row, _) in map.column(term.row) {
                    entries.try_reserve(1).map_err(|e| policy.reserve(e))?;
                    entries.push(Term { row, ..term });
                    c.completed(1).await?;
                }
            }
        }
        match Coefficients::from_entries(entries, count, columns, policy, c).await? {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        }
    };
    Ok(Ok(Array {
        shape,
        offset,
        degree,
        coefficients,
        assigned: Assigned::All,
    }))
}
