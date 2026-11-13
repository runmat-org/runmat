//! Powell-damped Lagrangian BFGS. A rejected numerical update retains the
//! previous certified model; it never clamps a supplied Hessian or alters f.
use crate::builtins::math::optim::common::{conic, ScalarCooperation};
use crate::{build_runtime_error, BuiltinResult};
use clarabel::algebra::CscMatrix;
use runmat_builtins::BuiltinCatalogIdentity;

type Matrix = CscMatrix<f64>;
mod secant;
pub(super) use secant::{between, LagrangianPoint, Secant};

fn storage<T>(length: usize, owner: BuiltinCatalogIdentity) -> BuiltinResult<Vec<T>> {
    runmat_value::exact_capacity_vec(length).map_err(|source| {
        build_runtime_error(format!(
            "{}: Lagrangian model allocation failed",
            owner.name
        ))
        .with_builtin(owner.name)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
    })
}

pub(super) async fn initial(
    n: usize,
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Matrix> {
    let extent = n.checked_add(1).ok_or_else(|| {
        build_runtime_error("Lagrangian model extent overflow")
            .with_builtin(owner.name)
            .build()
    })?;
    let mut pointers = storage(extent, owner)?;
    let mut rows = storage(n, owner)?;
    let mut values = storage(n, owner)?;
    pointers.push(0);
    for index in 0..n {
        rows.push(index);
        values.push(1.0);
        pointers.push(values.len());
        cooperation.completed(1).await?;
    }
    Ok(Matrix::new(n, n, pointers, rows, values))
}

async fn dot(
    left: &[f64],
    right: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<f64>> {
    let (mut sum, mut correction) = (0.0_f64, 0.0_f64);
    for (&left, &right) in left.iter().zip(right) {
        let term = left * right;
        let next = sum + term;
        if !next.is_finite() {
            return Ok(None);
        }
        correction += if sum.abs() >= term.abs() {
            (sum - next) + term
        } else {
            (term - next) + sum
        };
        sum = next;
        cooperation.completed(1).await?;
    }
    let value = sum + correction;
    Ok(value.is_finite().then_some(value))
}

pub(super) async fn update(
    previous: &Matrix,
    displacement: &[f64],
    lagrangian_change: &[f64],
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Matrix>> {
    let n = displacement.len();
    if previous.m != n || previous.n != n || lagrangian_change.len() != n {
        return Err(build_runtime_error("Lagrangian update dimensions disagree")
            .with_builtin(owner.name)
            .build());
    }
    let mut product = storage(n, owner)?;
    product.resize(n, 0.0);
    for (column, &step) in displacement.iter().enumerate() {
        for entry in previous.colptr[column]..previous.colptr[column + 1] {
            let row = previous.rowval[entry];
            let value = previous.nzval[entry];
            product[row] += value * step;
            if row != column {
                product[column] += value * displacement[row];
            }
            cooperation.completed(1).await?;
        }
        cooperation.completed(1).await?;
    }
    let Some(curvature) = dot(displacement, &product, cooperation).await? else {
        return Ok(None);
    };
    if curvature <= 0.0 {
        return Ok(None);
    }
    let Some(measured) = dot(displacement, lagrangian_change, cooperation).await? else {
        return Ok(None);
    };
    let theta = if measured >= 0.2 * curvature {
        1.0
    } else {
        0.8 * curvature / (curvature - measured)
    };
    if !(theta > 0.0 && theta <= 1.0 && theta.is_finite()) {
        return Ok(None);
    }
    let mut damped = storage(n, owner)?;
    for (&change, &bs) in lagrangian_change.iter().zip(&product) {
        let value = theta * change + (1.0 - theta) * bs;
        if !value.is_finite() {
            return Ok(None);
        }
        damped.push(value);
        cooperation.completed(1).await?;
    }
    let Some(damped_curvature) = dot(displacement, &damped, cooperation).await? else {
        return Ok(None);
    };
    if damped_curvature <= 0.0 {
        return Ok(None);
    }
    let first = curvature.sqrt();
    let second = damped_curvature.sqrt();
    for (bs, r) in product.iter_mut().zip(&mut damped) {
        *bs /= first;
        *r /= second;
        if !bs.is_finite() || !r.is_finite() {
            return Ok(None);
        }
        cooperation.completed(1).await?;
    }
    let length = n
        .checked_add(1)
        .and_then(|next| n.checked_mul(next))
        .map(|v| v / 2)
        .ok_or_else(|| {
            build_runtime_error("Lagrangian model extent overflow")
                .with_builtin(owner.name)
                .build()
        })?;
    let mut pointers = storage(n + 1, owner)?;
    let mut rows = storage(length, owner)?;
    let mut values = storage(length, owner)?;
    pointers.push(0);
    for column in 0..n {
        let mut entry = previous.colptr[column];
        let end = previous.colptr[column + 1];
        for row in 0..=column {
            let old = if entry < end && previous.rowval[entry] == row {
                let value = previous.nzval[entry];
                entry += 1;
                value
            } else {
                0.0
            };
            let value = old - product[row] * product[column] + damped[row] * damped[column];
            if !value.is_finite() {
                return Ok(None);
            }
            if value != 0.0 {
                rows.push(row);
                values.push(value);
            }
            cooperation.completed(1).await?;
        }
        pointers.push(values.len());
    }
    let candidate = Matrix::new(n, n, pointers, rows, values);
    // Exact arithmetic preserves PSD under this update, but rounded storage
    // must independently pass the shared original-matrix certificate.
    if conic::convexity::certify(&candidate, owner, cooperation).await? {
        Ok(Some(candidate))
    } else {
        Ok(None)
    }
}
