//! Constant matrix powers and compact identity construction.
use super::*;
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn identity(
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Array> {
    let n = shape.first().copied().unwrap_or(1);
    let mut pointers = runmat_value::exact_capacity_vec(
        n.checked_add(1)
            .ok_or_else(|| policy.invalid("identity pointer overflow"))?,
    )
    .map_err(|e| policy.storage(e))?;
    let mut rows = runmat_value::exact_capacity_vec(n).map_err(|e| policy.storage(e))?;
    let mut values = runmat_value::exact_capacity_vec(n).map_err(|e| policy.storage(e))?;
    pointers.push(0);
    for row in 0..n {
        rows.push(row);
        values.push(1.);
        pointers.push(row + 1);
        c.completed(1).await?;
    }
    let offset = Offset::Sparse(
        runmat_value::SparseTensor::new(n, n, pointers, rows, values)
            .map_err(|e| policy.invalid(e))?,
    );
    Ok(Array {
        shape,
        offset,
        degree: Degree::Constant,
        coefficients: Coefficients::Zero,
        assigned: Assigned::All,
    })
}
pub(super) async fn lower(
    mut base: Array,
    mut exponent: f64,
    shape: SharedShape,
    columns: usize,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Array> {
    let n = base.shape.first().copied().unwrap_or(1);
    if base.shape.get(1).copied().unwrap_or(1) != n
        || base.shape.iter().skip(2).any(|&d| d != 1)
        || !crate::builtins::common::shape::array_shapes_equal(&base.shape, &shape)
    {
        return Err(policy.invalid("matrix power requires original square geometry"));
    }
    if !matches!(base.assigned, Assigned::All) {
        for row in 0..base.count(policy)? {
            if !base.assigned.at(row, policy)? {
                return Ok(Err(Stop::Unassigned));
            }
            c.completed(1).await?;
        }
    }
    if exponent == 1. {
        base.shape = shape;
        return Ok(Ok(base));
    }
    if exponent != 0. && base.degree != Degree::Constant {
        return Ok(Err(Stop::NonAffine));
    }
    if exponent < 0. {
        if n == 0 {
            return Ok(Ok(identity(shape, policy, c).await?));
        }
        if matches!(base.offset, Offset::Uniform(_)) {
            return Ok(Err(Stop::NonFinite));
        }
        let value = evaluation::invoke(
            "mpower",
            &[base.offset.value(), Value::Num(exponent)],
            policy,
            c,
        )
        .await?;
        if !construct::finite_numeric(value.value(), policy, c).await? {
            return Ok(Err(Stop::NonFinite));
        }
        let offset = match Offset::from_value(value.value(), policy)? {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        };
        return Ok(Ok(Array {
            shape,
            offset,
            degree: Degree::Constant,
            coefficients: Coefficients::Zero,
            assigned: Assigned::All,
        }));
    }
    let mut result = identity(shape.clone(), policy, c).await?;
    while exponent >= 1. {
        if exponent % 2. == 1. {
            result = match result
                .matrix_product(base.clone(), shape.clone(), columns, policy, c)
                .await?
            {
                Ok(value) => value,
                Err(stop) => return Ok(Err(stop)),
            };
        }
        exponent = (exponent / 2.).floor();
        if exponent >= 1. {
            base = match base
                .clone()
                .matrix_product(base, shape.clone(), columns, policy, c)
                .await?
            {
                Ok(value) => value,
                Err(stop) => return Ok(Err(stop)),
            };
        }
        c.completed(1).await?;
    }
    Ok(Ok(result))
}
